DATAPATH = '../../../datasets/bodmas.npz'
SAVEDIR = "../../../models/bodmas/torch"
MODE = "binary"
N_CLASS = 2
DENSE = 4
EPOCHS = 20
BATCH_SIZE = 128
TARGET_LABEL = 0
SEED = 12

import argparse
import datetime
import json
import os, sys

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")
print(f"parent_dir_path: {parent_dir_path}")

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import tqdm

from dataset import get_train_test_loaders_bodmas
from models.simple import SimpleModel
from models.malconv import MalConv
from backdoor_helper import get_poison_batch

np.random.seed(SEED)

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_args():
    """
    Parse Arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Convert string values to appropriate types
    config['classes'] = int(config.get('classes', N_CLASS))
    config['epochs'] = int(config.get('epochs', EPOCHS))
    config['batch_size'] = int(config.get('batch_size', BATCH_SIZE))
    config['test_batch_size'] = int(config.get('test_batch_size', 128))
    config['is_backdoor'] = True if config.get('is_backdoor') == 1 else False
    config['target_label'] = int(config.get('target_label', TARGET_LABEL))
    
    return argparse.Namespace(**config)

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data = data.to(device).float()
            target = target.to(device).long()
            targets.append(target.detach().cpu().numpy())

            output = model(data)
        
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]

            correct += pred.eq(target.view(-1)).sum().item()
    logger.info(f"Testing loss: {test_loss/len(test_loader)}, \t Testing Accuracy: {correct /len(test_loader.dataset)}")
    return test_loss/len(test_loader), correct /len(test_loader.dataset)
    
def train(model, data_loader, device, log_interval=10, total_epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(total_epochs):
        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            print(data.shape)
            print(data.min(), data.max())
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                logger.info('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
        # optimizer.step()  # Update weights based on accumulated gradients
        logger.info('Iter [{}/{}]:\t Training Accuracy: {}/{} ({:.2f}%)\n'.format(
                    epoch+1, total_epochs, 
                    correct, len(data_loader.dataset),
                    100. * correct / len(data_loader.dataset)))
    model.eval()
    return model

def train_backdoor(model, data_loader, device, log_interval=10, total_epochs=10, target_label=1, alpha_loss=0.5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(total_epochs):
        correct = 0
        total_l = 0
        poison_data_count = 0
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            # get the poisoned batch, we will poison a subset of this batch 
            # while keeping the remaining unchanged
            data, target, poison_cnt = get_poison_batch(batch, target_label, device)
            poison_data_count += poison_cnt
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_l += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check backdoor accuracy
            loss.backward()
            optimizer.step()
            if epoch % log_interval == 0:
                logger.info('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
        total_l /= batch_idx
        # optimizer.step()  # Update weights based on accumulated gradients
        logger.info('Iter [{}/{}]:\t Training Accuracy: {}/{} ({:.2f}%)\n'.format(
                    epoch+1, total_epochs, 
                    correct, len(data_loader.dataset),
                    100. * correct / len(data_loader.dataset)))
        logger.info(
            '___PoisonTrain {} , epoch {:3d}, Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch,
                                                                            total_l, correct, 
                                                                            len(data_loader.dataset),
                                                                            100. * correct / len(data_loader.dataset), 
                                                                            poison_data_count))
    model.eval()
    return model

def test_backdoor(model, test_loader, device, target_label=1):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(test_loader)):
            data, targets, poison_num = get_poison_batch(batch, target_label, device, adversarial_index=-1, evaluation=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    model.train()
    return total_l, acc, correct, poison_data_count

if __name__ == "__main__":
    current_time = str(datetime.datetime.now())
    #
    # Get and validate arguments
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    os.makedirs(args.savedir, exist_ok=True)
    print("\nLoading Data ...")
    train_dl, valid_dl = get_train_test_loaders_bodmas(args.datapath, args.batch_size, 
                                                args.test_batch_size)
    # num_channels = valid_dl.dataset[0][0].shape[0]
    input_shape = valid_dl.dataset[0][0].shape[0]
    name = "none" if args.dense == 0 else args.dense
    args.name = f"malware_bodmas_binary_scaled_{name}-2"
    #
    # Training
    #
    print("\n-------- Training Parameters --------- ")
    print(f"NAME: {args.name}")
    print(f"DENSE: {args.dense}")
    print(f"EPOCHS: {args.epochs}")
    print(f"SEED: {SEED}")
    # model = SimpleModel(input_shape, DENSE)
    model = MalConv()
    model = model.to(device)
    if args.is_backdoor:
        print("\n--------Backdoor Training --------- ")
        model = train_backdoor(model, train_dl, device, target_label=args.target_label)
        os.makedirs(f"{args.savedir}/{args.model}/backdoor/", exist_ok=True)
        torch.save(model, f"{args.savedir}/{args.model}/{current_time}.pth")
        print("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
        print("\n--------Backdoor Testing --------- ")
        _, test_acc = test_backdoor(model, valid_dl, device, args.target_label)
    else:
        print("\n--------Normal Training --------- ")
        model = train(model, train_dl, device)
        os.makedirs(f"{args.savedir}/{args.model}", exist_ok=True)
        torch.save(model, f"{args.savedir}/{args.model}/{current_time}.pth")
        print("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
    
    
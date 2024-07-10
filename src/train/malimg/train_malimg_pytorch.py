"""
Script to train Malimg image malware classifiers and save to .h5 file for verification.

NOTES:
models = [linear-2, 4-2, 16-2]

"""
import json
import os, sys
import datetime

from termcolor import colored

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")
print(f"parent_dir_path: {parent_dir_path}")

# import tensorflow as tf
# from tensorflow.keras import layers, Model, Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torchvision.utils as vutils

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

import logging

from dataset import get_train_test_loaders
from models.cnn import CNN
from models.mobilenet import MobileNetV2
from models.resnet_bak import ResNet18
from models.resnet import ResNet
from backdoor_helper import get_poison_batch, set_seed
from ft_dataset import load_data_loaders, pre_split_dataset

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

from utils import logger

import torch
import torch.nn as nn
from tqdm import tqdm

DATAPATH = "../../../datasets/malimg"
SAVEDIR = "../../../models/malimg/torch"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
SEED = 12
TARGET_LABEL = 0

LOG_PATH = "models"
# tf.random.set_seed(SEED) # Sets seed for training
np.random.seed(SEED)
stdoutOrigin=sys.stdout
# def get_args():
#     """
#     Parse Arguments.
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-c", "--classes", type=int, default=N_CLASS, help="total number of class for nullclass mode.")
#     parser.add_argument("-d", "--datapath", type=str, default=DATAPATH)
#     parser.add_argument("-n", "--name", type=str, default = None)
#     parser.add_argument("--savedir", type=str, default=SAVEDIR)
#     parser.add_argument("--model", type=str, default="mobilenetv2")
    
#     # Training Parameters
#     parser.add_argument("-s", "--imsize", type=int, default=IMSIZE)
#     parser.add_argument("-c1", "--conv1", type=int, default=CONV1)
#     parser.add_argument("-e", "--epochs", type=int, default=EPOCHS)
#     parser.add_argument("-bs", "--batch_size", type=int, default=BATCH_SIZE)
#     parser.add_argument("-lr", "--lr", type=float, default=0.001)
    
#     args = parser.parse_args()
#     return args


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
    config['imsize'] = int(config.get('imsize', IMSIZE))
    config['conv1'] = int(config.get('conv1', CONV1))
    config['epochs'] = int(config.get('epochs', EPOCHS))
    config['batch_size'] = int(config.get('batch_size', BATCH_SIZE))
    config['test_batch_size'] = int(config.get('test_batch_size', 128))
    config['is_backdoor'] = True if config.get('is_backdoor') == 1 else False
    config['target_label'] = int(config.get('target_label', TARGET_LABEL))
    config['num_poison'] = int(config.get('num_poison', 4))

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
    logger.info(colored(f"[Clean] Testing loss: {test_loss/len(test_loader)}, \t Testing Accuracy: {correct /len(test_loader.dataset)}", "green"))
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
            output = model(data)
            # loss = criterion(output, target)
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

#----------*----------#
#------BACKDOOR-------#
#----------*----------#

def train_backdoor(model, data_loader, device, log_interval=10, 
                   total_epochs=10, target_label=1, 
                   num_poison=4, model_name="mobilenetv2", 
                   lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(total_epochs):
        correct = 0
        total_l = 0
        poison_data_count = 0
        for batch_idx, batch in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            # get the poisoned batch, we will poison a subset of this batch 
            # while keeping the remaining unchanged
            # data, target, poison_cnt = get_poison_batch(batch, target_label, 
            #                                             device, 
            #                                             poisoning_per_batch=num_poison)
            data, target = batch
            poison_cnt = 0
            # if model_name == "resnet":
            #     data = data.repeat(1, 3, 1, 1)
            # print(data.shape)
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
            '___PoisonTrain, epoch {:3d}, Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(epoch,
                                                                            total_l, correct, 
                                                                            len(data_loader.dataset),
                                                                            100. * correct / len(data_loader.dataset), 
                                                                            poison_data_count))
    model.eval()
    return model

def test_backdoor(model, test_loader, device, 
                  target_label=1):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(test_loader)):
            clean_data, tgt = batch
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
    
    batch_img = torch.cat(
    [clean_data[:8].clone().cpu(), data[:8].clone().cpu()], 0)
    grid = vutils.make_grid(batch_img, nrow=8, padding=2, normalize=True)
    parent_dir = f"../../../models/bodmas/backdoor_imgs"
    # save style-transferred images, just for visualization
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_path = f"{parent_dir}/backdoor_output_grid.png"
    vutils.save_image(grid, output_path)
    # model.train()
    logger.info(colored(f"[Backdoor] Testing loss: {total_l}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {poison_data_count}", "red"))
    return total_l, acc, correct, poison_data_count


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproduce Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
if __name__ == "__main__":
    set_seed(SEED)
    current_time = str(datetime.datetime.now())
    # get the start time for saving trained model
    args = get_args()
    # sys.stdout = open(f"{LOG_PATH}/{args.dataset}/torch/{args.model}/logs/{str(datetime.datetime.now())}.log", "a+")
    os.makedirs(args.savedir, exist_ok=True)
    logger.info(args)
    convs_sizes = [2, 8, 32]
    #
    # Generate Training Set
    #
    # print("\nLoading Data ...")
    # train_dl, valid_dl = get_train_test_loaders(args.datapath, args.batch_size, 
    #                                             args.test_batch_size, args.imsize)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for s in convs_sizes:
    # args.conv1 = s
    
    parent_p = "../../../datasets/malimg_ft"
    pre_split_dataset(args.datapath, args.batch_size, 
                      args.test_batch_size, 
                      args.imsize, args.ft_size)
    
    # logging.basicConfig(filename=f"../../../models/{args.dataset}/{args.model}/logs/{str(datetime.datetime.now())}_temp_all_samples.log", filemode="w", format="%(name)s → %(levelname)s: %(message)s")

    train_dl, poison_dl, valid_dl, ft_dl, _ = load_data_loaders(data_path=parent_p,
                                                  ft_size=args.ft_size,
                                                  batch_size=args.batch_size, 
                                                  test_batch_size=args.test_batch_size,
                                                  num_workers=56, dataset=args.dataset,
                                                  target_label=args.target_label,
                                                  poison_rate=args.poison_rate)
    num_channels = valid_dl.dataset[0][0].shape[0]
    print(f"num_channels: {num_channels}")
    # train_dl, valid_df, ft_dl = get_train_test_ft_loaders(args.datapath, args.batch_size, 
    #                                                     args.test_batch_size, args.imsize, 
    #                                                     args.ft_size)
    
    name = "linear" if args.conv1 == 0 else args.conv1
    args.name = f"malware_malimg_family_scaled_{name}-25"
    # -----*------ #
    # --Training-- #
    # -----*------ #
    print("\n-------- Training Parameters --------- ")
    print(f"NAME: {args.name}")
    print(f"IMSIZE: {args.imsize}")
    print(f"CONV1: {args.conv1}")
    print(f"EPOCHS: {args.epochs}")
    print(f"SEED: {SEED}")
    
    print("\n-------- Training --------- ")
    # model = train_family_classifier(X_train, X_val, y_train, y_val, args)
    # model = CNN(args.imsize, num_channels, args.conv1, args.classes)
    if args.model == "cnn":
        model = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_channels, args.classes)
    elif args.model == "resnet":
        model = ResNet(n_classes=args.classes, input_shape=[1, 64, 64])
    else:
        pass
    model.to(device)
    file_to_save = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    if args.is_backdoor:
        print("\n--------Backdoor Training --------- ")
        model = train_backdoor(model, poison_dl, device, 
                               target_label=args.target_label,
                               total_epochs=args.epochs, 
                               num_poison=args.num_poison, 
                               model_name=args.model,
                               lr = args.lr)
        os.makedirs(f"{args.savedir}/{args.model}/backdoor", exist_ok=True)
        torch.save(model.state_dict(), f"{args.savedir}/{args.model}/backdoor/{file_to_save}.pth")
        logger.info(colored(f"Saved model at {f'{args.savedir}/{args.model}/backdoor/{file_to_save}.pth'}", "blue"))
        print("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
        print("\n--------Backdoor Testing --------- ")
        total_l, acc, correct, poison_data_count = test_backdoor(model, valid_dl, device, 
                                                                 args.target_label)
        model.to("cpu")
    else:
        print("\n--------Normal Training --------- ")
        model = train(model, train_dl, device)
        os.makedirs(f"{args.savedir}/{args.model}", exist_ok=True)
        torch.save(model.state_dict(), f"{args.savedir}/{args.model}/{file_to_save}.pth")
        print("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# General Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
# if __name__ == "__main__":
#     #
#     # Get and validate arguments
#     #
#     args = get_args()
#     if args.name == None:
#         name = "none" if args.conv1 == 0 else args.conv1
#         args.name = f"malware_malimg_family_scaled_{name}-25"    #
#     # Training
#     #
#     print("\n-------- Training Parameters --------- ")
#     print(f"NORMALIZE: {not args.no_normalize}")
#     print(f"NAME: {args.name}")
#     print(f"IMSIZE: {args.imsize}")
#     print(f"CONV1: {args.conv1}")
#     print(f"EPOCHS: {args.epochs}")
    
#     print("\n-------- Training --------- ")

#     # train_family_classifier(args)

        

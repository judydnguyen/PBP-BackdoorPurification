import copy
import gc
import json
import sys, os
import math

from termcolor import colored
from tqdm import tqdm

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from malimg.train_malimg_pytorch import test, test_backdoor
from models.cnn import CNN
from models.resnet import ResNet
from models.mobilenet import MobileNetV2
from backdoor_helper import set_seed
from utils import *
from plot_utils import *
from ft_dataset import load_data_loaders
from defense_helper import add_noise, get_grad_mask_by_layer, masked_feature_shift_loss

import argparse
from pprint import  pformat

import torch
import logging
import torch.nn as nn
from torch import optim

import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

# from utils.aggregate_block.save_path_generate import generate_save_folder
# from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
# from utils.aggregate_block.fix_random import fix_random
# from utils.aggregate_block.dataset_and_transform_generate_ft import dataset_and_transform_generate
# from utils.bd_dataset import prepro_cls_DatasetBD

# from utils.aggregate_block.model_trainer_generate import generate_cls_model
# from load_data import CustomDataset, CustomDataset_v2

SEED = 12
set_seed(SEED)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to JSON config file.")
    args, _ = parser.parse_known_args()  # Parse only known arguments
    with open(args.config, 'r') as f:
        config_from_file = json.load(f)
    
    # Load default arguments
    default_config = {
        'device': None,
        'ft_mode': 'all',
        'num_classes': 25,
        'attack_label_trans': 'all2one',
        'pratio': None,
        'epochs': None,
        'dataset': None,
        'dataset_path': '../data',
        'folder_path': '../models',
        'target_label': 0,
        'batch_size': 128,
        'lr': None,
        'random_seed': 0,
        'model': None,
        'split_ratio': None,
        'log': False,
        'initlr': None,
        'pre': False,
        'save': False,
        'linear_name': 'linear',
        'lb_smooth': None,
        'alpha': 0.2
    }
    
    # Update default arguments with values from config file
    default_config.update(config_from_file)
    
    # Add arguments to the parser
    for key, value in default_config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser

def get_optimizer(net, ft_mode, linear_name, f_lr):
    if ft_mode == 'fe-tuning':
        init = True
        log_name = 'FE-tuning'
    elif ft_mode == 'ft-init':
        init = True
        log_name = 'FT-init'
    elif ft_mode == 'ft':
        init = False
        log_name = 'FT'
    elif ft_mode == 'lp':
        init = False
        log_name = 'LP'
    elif ft_mode == 'fst':
        init = True
        log_name = 'FST'
    elif ft_mode == 'proposal':
        init = True
        log_name = 'proposal'
    else:
        raise NotImplementedError('Not implemented method.')

    param_list = []
    for name, param in net.named_parameters():
        if linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if ft_mode == 'lp':
            if linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif ft_mode in ['ft', 'fst', 'ft-init', 'proposal']:
            param.requires_grad = True
            param_list.append(param)
        elif ft_mode == 'fe-tuning':
            if linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
                
    optimizer = optim.Adam(param_list, lr=f_lr)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def finetune(net, optimizer, criterion,
             ft_dl, test_dl, f_epochs=1, 
             ft_mode="ft", device="cuda", logger=None,
             logging_path="path/to/log", lbs_criterion=None,
             args=None, weight_mat_ori=None, 
             original_linear_norm=0):
    # At the start of main(), before the training loop
    # net.train()
    # if ft_mode == 'proposal':
    #     net = add_noise(net, device, stddev=args.stddev)
    os.makedirs(logging_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{logging_path}/log/mode_{ft_mode}')
    print("Fine-tuning mode:", ft_mode)
    cur_clean_acc, cur_adv_acc = 0.0, 0.0
    writer.add_scalar('Validation Clean ACC', cur_clean_acc, 0)
    writer.add_scalar('Validation Backdoor ACC', cur_adv_acc, 0)
    
    log_path = f"{logging_path}/plots/mode_{ft_mode}"
    
    # original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    # weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')
    
    # if ft_mode == "proposal":
    #     net_cpy = copy.deepcopy(net)
    #     optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
    #     mask = get_grad_mask_by_layer(net_cpy, optimizer_cp, ft_dl, device=device, layer=args.linear_name)
    #     del net_cpy, optimizer_cp

    if ft_mode == 'proposal':
        net = add_noise(net, device, stddev=args.stddev)
        
    for epoch in range(1, f_epochs+1):
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
        logging.info(f'Epoch: {epoch}')
        if ft_mode == "proposal":
            net_cpy = copy.deepcopy(net)
            optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
            mask = get_grad_mask_by_layer(net_cpy, optimizer_cp, ft_dl, device=device, layer=args.linear_name)
            del net_cpy, optimizer_cp
        # net.train()
        for batch_idx, (x, labels) in tqdm(enumerate(ft_dl)):
            optimizer.zero_grad()
            x, labels = x.to(device), labels.to(device)
            log_probs = net(x)
            
            if lbs_criterion is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if ft_mode == 'fst':
                    constraint_loss = torch.norm(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)
                    ce_loss = criterion(log_probs, labels.long())
                    print(f"constraint_loss: {constraint_loss}, \tCE_loss: {ce_loss}")
                    loss = constraint_loss*args.alpha + ce_loss
                    # import IPython
                    # IPython.embed()
                    
                elif ft_mode == 'proposal':
                    # try new loss
                    loss = masked_feature_shift_loss(eval(f'net.{args.linear_name}.weight'), weight_mat_ori, mask) + criterion(log_probs, labels.long())
                else:
                    loss = criterion(log_probs, labels.long()).mean()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            exec(exec_str)

            _, predicted = torch.max(log_probs, -1)
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)
        
        
        gc.collect()
        # scheduler.step()
        one_epoch_loss = sum(batch_loss_list)

        logging.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logging.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info('-------------------------------------')
        
        writer.add_scalar('Training Loss', one_epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', train_correct/train_tot, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
        
        logger.info(colored(f"Start validation for current epoch: [{epoch}/{args.f_epochs}]\n", "blue"))
        print("\n--------Normal Testing --------- ")
        loss_c, acc_c = test(net, test_dl, device)
        print("\n--------Backdoor Testing --------- ")
        loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, test_dl, device, 
                                                                    args.target_label)
        writer.add_scalar('Validation Clean ACC', acc_c, epoch)
        writer.add_scalar('Validation Backdoor ACC', acc_bd, epoch)
        
        metric_info = {
            f'clean acc': acc_c,
            f'clean loss': loss_c,
            f'backdoor acc': acc_bd,
            f'backdoor loss': loss_bd
        }
        cur_clean_acc = metric_info['clean acc']
        cur_adv_acc = metric_info['backdoor acc']
        logging.info('*****************************')
        logging.info(colored(f'Fine-tunning mode: {ft_mode}', "green"))
        logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
        logging.info('*****************************')
                
        ori_weights = weight_mat_ori.detach().cpu().numpy()
        current_weights = eval(f'net.{args.linear_name}.weight').detach().cpu().numpy()
        plot_last_w(ori_weights, current_weights, epoch, log_path)
        
    # ori_weights = weight_mat_ori.detach().cpu().numpy()
    # current_weights = eval(f'net.{args.linear_name}.weight').detach().cpu().numpy()
    
    # plot_last_w(ori_weights, current_weights)
    # Assuming ori_weights and current_weights are your original 1D weight arrays
    net.eval()
    writer.close()
    
def main():
    ### 1. config args, save_path, fix random seed
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add arguments using the add_args function
    parser = add_args(parser)

    # Parse the arguments
    args = parser.parse_args()

    # Now you can access the arguments using dot notation
    print("Device:", args.device)
    print("Number of classes:", args.num_classes)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    
    # ------------ Loading a pre-trained (backdoored) model -------------- #
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_to_load = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    file_to_load = f'{args.folder_path}/backdoor/{file_to_load}.pth'
        
    # logFormatter = logging.Formatter(
    #     fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
    #     datefmt='%Y-%m-%d:%H:%M:%S',
    # )
    # logger = logging.getLogger()

    # if args.log:
    #     fileHandler = logging.FileHandler(args.save_path + '.log')
    #     fileHandler.setFormatter(logFormatter)
    #     logger.addHandler(fileHandler)


    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)

    # logger.setLevel(logging.INFO)
    # logging.info(pformat(args.__dict__))

    ### 2. set the clean train data and clean test data
    # train_dl, test_dl, ft_dl = get_train_test_ft_loaders(args.datapath, args.batch_size, 
    #                                                      args.test_batch_size, args.imsize, 
    #                                                      args.ft_size)
    parent_p = "../../../datasets/malimg_ft"
    train_dl, _, test_dl, ft_dl, _ = load_data_loaders(data_path=parent_p,
                                                  ft_size=args.ft_size,
                                                  batch_size=args.batch_size, 
                                                  test_batch_size=args.test_batch_size,
                                                  num_workers=56, 
                                                  dataset=args.dataset,
                                                  poison_rate=args.poison_rate,
                                                  target_label=args.target_label)

    ### 3. get model
    num_channels = test_dl.dataset[0][0].shape[0]

    if args.model == "cnn":
        net = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "mobilenetv2":
        net = MobileNetV2(num_channels, args.classes)
    elif args.model == "resnet":
        net = ResNet(n_classes=args.classes, input_shape=[1, 64, 64])
    else:    
        raise NotImplementedError(f"{args.model} is not supported")

    state_dict = torch.load(file_to_load)

    # Load the state dictionary into the model
    net.load_state_dict(state_dict)
    logger.info(colored(f"Loaded model at {file_to_load}", "blue"))
    net.to(device)
    
    # ori_net = copy.deepcopy(net)
    # ori_net.eval()
    
    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')
    
    print("\n--------Normal Testing --------- ")
    loss_c, acc_c = test(net, test_dl, device)
    print("\n--------Backdoor Testing --------- ")
    loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, test_dl, device, 
                                                                args.target_label)
    metric_info = {
        f'clean acc': acc_c,
        f'clean loss': loss_c,
        f'backdoor acc': acc_bd,
        f'backdoor loss': loss_bd
    }
    cur_clean_acc = metric_info['clean acc']
    cur_adv_acc = metric_info['backdoor acc']

    logging.info('*****************************')
    logging.info(f"Load from {args.folder_path}")
    # logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logging.info('Original performance')
    logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logging.info('*****************************')

    # ---------- Start Fine-tuning ---------- #
    logging_path = f'{args.log_dir}/target_{args.target_label}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}'
    # ft_modes = ['ft', 'ft-init', 'fe-tuning', 'lp', 'fst', 'proposal']
    ft_modes = ['proposal']
    for ft_mode in ft_modes:
        # model = copy.deepcopy(ori_net)
        # model.load_state_dict(state_dict)
        # model.to(device)
        net.load_state_dict(state_dict)
        net.to(device)
        print("\n--------Backdoor Testing --------- ")
        loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, test_dl, device, 
                                                                    args.target_label)
        # model.train()
        optimizer, criterion = get_optimizer(net, ft_mode, args.linear_name, args.f_lr)
        model = finetune(net, optimizer, criterion, ft_dl, test_dl, args.f_epochs, 
                      ft_mode, device, logger, logging_path, args=args, 
                      weight_mat_ori=weight_mat_ori, 
                      original_linear_norm=original_linear_norm)
        if args.save:
            model_save_path = f'{args.folder_path}/target_{args.target_label}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}/mode_{ft_mode}'
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{model_save_path}/checkpoint.pt')
        del model
if __name__ == '__main__':
    main()
    
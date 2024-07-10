"""
Script to train Malimg image malware classifiers and save to .h5 file for verification.

NOTES:
models = [linear-2, 4-2, 16-2]

"""
import copy
import json
import os, sys
import datetime

from termcolor import colored


# Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")
sys.path.append("../../")

import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
import torchvision.utils as vutils

import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

from explainable_backdoor_utils import get_backdoor_data
from models.cnn import CNN
from models.mobilenet import MobileNetV2
from models.resnet_bak import ResNet18
from models.embernn import EmberNN
from models.simple import DeepNN, SimpleModel
from models.CNN_Models import CNNMalware_Model1
from models.ANN_Models import ANNMalware_Model1, MalConv
from utils import final_evaluate, logger
from attack_utils import load_wm

from backdoor_helper import set_seed
from ft_dataset import get_backdoor_loader, load_data_loaders, pre_split_dataset_ember, separate_test_data
from common_helper import load_np_data

import torch
import torch.nn as nn
from tqdm import tqdm

DATAPATH = "datasets/ember"
DESTPATH = "datasets/ember/np"
SAVEDIR = "models/malimg/torch"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
SEED = 12
TARGET_LABEL = 0

# tf.random.set_seed(SEED) # Sets seed for training
np.random.seed(SEED)

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
    config['subset_family'] = "kyugo"

    return argparse.Namespace(**config)

def create_backdoor(model, device):
    # wm_config = load_wm(DESTPATH)
    model.to(device)
    model.train()
    model.load_state_dict(torch.load("models/ember/torch/embernn/tgt_0_epochs_5_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"))
    generate_backdoor_data(model, device)

#----------*----------#
#------BACKDOOR-------#
#----------*----------#

def generate_backdoor_data(model, device):
    X_train_loaded, y_train_loaded, X_val, y_val , _ = load_np_data(DATAPATH)
    logger.info(colored(f"Start generating backdoor sampled with set of size: {X_train_loaded.shape[0]}", "red"))
    X_train_watermarked, y_train_watermarked, X_test_mw = get_backdoor_data(X_train_loaded, y_train_loaded, X_val, y_val, 
                                                                            copy.deepcopy(model), device, DESTPATH)
    logger.info(colored(f"Size of training backdoor data: {X_train_watermarked.shape[0]}"))
    
    backdoored_X, backdoored_y = torch.from_numpy(X_train_watermarked), torch.from_numpy(y_train_watermarked)
    backdoored_dataset = TensorDataset(backdoored_X, backdoored_y)
    backdoored_loader = DataLoader(backdoored_dataset, batch_size=512, shuffle=True, num_workers=54)
    return backdoored_loader

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproduce Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
if __name__ == "__main__":
    set_seed(SEED)
    current_time = str(datetime.datetime.now())
    # get the start time for saving trained model
    args = get_args()
    os.makedirs(args.savedir, exist_ok=True)
    convs_sizes = [2, 8, 32]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parent_p = "../../../datasets/ember"
    pre_split_dataset_ember(args.datapath, args.ft_size, SEED, parent_p)

    train_dl, backdoor_dl, valid_dl, ft_dl, \
        backdoor_test_dl, X_test_loaded, y_test_loaded, X_subset_trojaned = load_data_loaders(data_path=parent_p,
                                                ft_size=args.ft_size,
                                                batch_size=args.batch_size, 
                                                test_batch_size=args.test_batch_size,
                                                num_workers=56, val_size=0,
                                                poison_rate=args.poison_rate,
                                                dataset=args.dataset)
    
    X_test_remain_mal, X_test_benign = separate_test_data(X_test_loaded, y_test_loaded)
    
    num_channels = valid_dl.dataset[0][0].shape[0]
    logger.info(f"Num channels: {num_channels}")
    
    name = "linear" if args.conv1 == 0 else args.conv1
    args.name = f"malware_malimg_family_scaled_{name}-25"
    # -----*------ #
    # --Training-- #
    # -----*------ #
    
    # logger.info("\n-------- Training --------- ")
    if args.model == "cnn":
        model = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "simple":
        model = SimpleModel(num_channels, 16)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_channels, args.classes)
    elif args.model == "resnet":
        model = ResNet18(num_classes=args.classes)
    elif args.model == "embernn":
        model = EmberNN(num_channels)
    else:
        pass
    model.to(device)
    file_to_save = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    # model.load_state_dict(torch.load("models/ember/torch/embernn/tgt_0_epochs_10_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"))
    model_save_path = f"{args.savedir}/{args.model}" if not args.is_backdoor else f"{args.savedir}/{args.model}/backdoor"
    create_backdoor(model, device)
    print(f"Successfully created backdoor data for {args.model} model")        
    
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
#     logger.info("\n-------- Training Parameters --------- ")
#     logger.info(f"NORMALIZE: {not args.no_normalize}")
#     logger.info(f"NAME: {args.name}")
#     logger.info(f"IMSIZE: {args.imsize}")
#     logger.info(f"CONV1: {args.conv1}")
#     logger.info(f"EPOCHS: {args.epochs}")
    
#     logger.info("\n-------- Training --------- ")

#     # train_family_classifier(args)

        
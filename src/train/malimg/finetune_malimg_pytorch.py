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
from models.mobilenet import MobileNetV2
from backdoor_helper import set_seed
from utils import *
from plot_utils import *
from ft_dataset import load_data_loaders
from defense_helper import get_grad_mask_by_layer, masked_feature_shift_loss

import argparse
from pprint import  pformat

import torch
import torch.nn.functional as F
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

def cmi_penalty(y, z_mu, z_sigma, reference_params):
    num_samples = y.shape[0]
    dimension = reference_params.shape[1] // 2
    # if self.ds_bundle.name == 'py150':
    #     is_labeled = ~torch.isnan(y)
    #     flattened_y = y[is_labeled]
    #     z_mu = z_mu[is_labeled.view(-1)]
    #     z_sigma = z_sigma[is_labeled.view(-1)]
    #     target_mu = self.reference_params[flattened_y.to(dtype=torch.long), :dimension]
    #     target_sigma = F.softplus(self.reference_params[flattened_y.to(dtype=torch.long), dimension:])
    # else:
    target_mu = reference_params[y.to(dtype=torch.long), :dimension]
    target_sigma = F.softplus(reference_params[y.to(dtype=torch.long), dimension:])
    cmi_loss = torch.sum((torch.log(target_sigma) - torch.log(z_sigma) + (z_sigma ** 2 + (target_mu - z_mu) ** 2) / (2*target_sigma**2) - 0.5)) / num_samples
    return cmi_loss

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
        'attack_target': 0,
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

def main():
    ### 1. config args, save_path, fix random seed
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add arguments using the add_args function
    parser = add_args(parser)

    # Parse the arguments
    args = parser.parse_args()
    
    # At the start of main(), before the training loop
    logging_path = f'../../../logging/ft/target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-f_epochs_{args.f_epochs}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}'
    os.makedirs(logging_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{logging_path}/mode_{args.ft_mode}')
    

    # Now you can access the arguments using dot notation
    print("Device:", args.device)
    print("Fine-tuning mode:", args.ft_mode)
    print("Number of classes:", args.num_classes)
    # args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    # args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    # fix_random(args.random_seed)
    
    if args.lb_smooth is not None:
        lbs_criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.lb_smooth)
    
    if args.ft_mode == 'fe-tuning':
        init = True
        log_name = 'FE-tuning'
    elif args.ft_mode == 'ft-init':
        init = True
        log_name = 'FT-init'
    elif args.ft_mode == 'ft':
        init = False
        log_name = 'FT'
    elif args.ft_mode == 'lp':
        init = False
        log_name = 'LP'
    elif args.ft_mode == 'fst':
        assert args.alpha is not None
        init = True
        log_name = 'FST'
    elif args.ft_mode == 'proposal':
        assert args.alpha is not None
        init = True
        log_name = 'proposal'
    else:
        raise NotImplementedError('Not implemented method.')
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_to_load = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    args.folder_path = f'{args.folder_path}/backdoor/{file_to_load}.pth'
    
    # if not args.pre:
    #     args.folder_path = f'../record_{args.dataset}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-initlr_{args.initlr}'
    #     os.makedirs(f'../logs_{args.model}_{args.dataset}/{log_name}/{args.attack}', exist_ok=True)
    #     args.save_path = f'../logs_{args.model}_{args.dataset}/{log_name}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
    # else:
    #     args.folder_path = f'../record_{args.dataset}_pre/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-initlr_{args.initlr}'
    #     os.makedirs(f'../logs_{args.model}_{args.dataset}_pre/{log_name}/{args.attack}', exist_ok=True)
    #     args.save_path = f'../logs_{args.model}_{args.dataset}_pre/{log_name}/{args.attack}/' + f'pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
        
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    if args.log:
        fileHandler = logging.FileHandler(args.save_path + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)


    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))


    ### 2. set the clean train data and clean test data
    # train_dl, test_dl, ft_dl = get_train_test_ft_loaders(args.datapath, args.batch_size, 
    #                                                      args.test_batch_size, args.imsize, 
    #                                                      args.ft_size)
    parent_p = "../../../datasets/malimg_ft"
    train_dl, test_dl, ft_dl = load_data_loaders(data_path=parent_p,
                                                  ft_size=args.ft_size,
                                                  batch_size=args.batch_size, 
                                                  test_batch_size=args.test_batch_size,
                                                  num_workers=56)

    ### 3. get model
    num_channels = test_dl.dataset[0][0].shape[0]
    print(f"num_channels: {num_channels}")
    if args.model == "cnn":
        net = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "mobilenetv2":
        net = MobileNetV2(num_channels, args.classes)
    else:    
        raise NotImplementedError(f"{args.model} is not supported")

    state_dict = torch.load(args.folder_path)

    # Load the state dictionary into the model
    net.load_state_dict(state_dict)
    logger.info(colored(f"Loaded model at {args.folder_path}", "blue"))
    net.to(device)
    
    ori_net = copy.deepcopy(net)
    ori_net.eval()
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
    logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logging.info('Original performance')
    logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logging.info('*****************************')


    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')

    param_list = []
    for name, param in net.named_parameters():
        if args.linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if args.ft_mode == 'lp':
            if args.linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif args.ft_mode in ['ft', 'fst', 'ft-init', 'proposal']:
            param.requires_grad = True
            param_list.append(param)
        elif args.ft_mode == 'fe-tuning':
            if args.linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        
        

    # optimizer = optim.SGD(param_list, lr=args.f_lr, momentum = 0.9)
    optimizer = optim.Adam(param_list, lr=args.f_lr)
    criterion = nn.CrossEntropyLoss()
    
    writer.add_scalar('Validation Clean ACC', cur_clean_acc, 0)
    writer.add_scalar('Validation Backdoor ACC', cur_adv_acc, 0)
    
    log_path = f"../../../plots/{args.ft_mode}"
    # net_cpy = copy.deepcopy(net)
    # optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
    # mask = get_grad_mask_by_layer(net_cpy, optimizer_cp, ft_dl, device=device)
    
    for epoch in range(1, args.f_epochs+1):
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        
        logging.info(f'Epoch: {epoch}')
        net_cpy = copy.deepcopy(net)
        optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
        mask = get_grad_mask_by_layer(net_cpy, optimizer_cp, ft_dl, device=device)
        net.train()
        for batch_idx, (x, labels) in tqdm(enumerate(ft_dl)):
            optimizer.zero_grad()
            x, labels = x.to(device), labels.to(device)
            feat_x = net.features(x)
            ori_feat_x = ori_net.features(x)
            log_probs = net(x)
            ori_log_probs = ori_net(x)
            if args.lb_smooth is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if args.ft_mode == 'fst':
                    loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels.long())
                elif args.ft_mode == 'proposal':
                    # kd_term = kd_loss(log_probs, labels, ori_log_probs)
                    # entropy_term = entropy_loss(log_probs)
                    # # entropy_term_2 = entropy_loss(ori_log_probs)
                    # # cosine_similarity_term = F.cosine_similarity(eval(f'net.{args.linear_name}.weight'), weight_mat_ori, dim=1)
                    # # print(f"entropy_term: {entropy_term}")
                    # loss = -torch.sum(feat_x - ori_feat_x)*args.alpha + criterion(log_probs, labels.long()).mean()
                    # # loss = kd_term
                    
                    # fedsr
                    # s_loss = CosineSimilarityLoss()
                    # loss_stat = s_loss(feat_x, ori_feat_x)
                    # print(f"loss_stat: {loss_stat}")
                    # loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels.long()) - 0.2*loss_stat
                    
                    # try new loss
                    loss = masked_feature_shift_loss(eval(f'net.{args.linear_name}.weight'), weight_mat_ori, mask)*0.1 + criterion(log_probs, labels.long())
                    
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
        
        
        del net_cpy, optimizer_cp
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
        logging.info(f'Fine-tunning mode: {args.ft_mode}')
        logging.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
        logging.info('*****************************')
                
        ori_weights = weight_mat_ori.detach().cpu().numpy()
        current_weights = eval(f'net.{args.linear_name}.weight').detach().cpu().numpy()
        plot_last_w(ori_weights, current_weights, epoch, log_path)
        
    # ori_weights = weight_mat_ori.detach().cpu().numpy()
    # current_weights = eval(f'net.{args.linear_name}.weight').detach().cpu().numpy()
    
    # plot_last_w(ori_weights, current_weights)
    # Assuming ori_weights and current_weights are your original 1D weight arrays


    writer.close()
    if args.save:
        model_save_path = f'defense_results/{args.attack}/pratio_{args.pratio}-target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}-sratio_{args.split_ratio}-lr_{args.lr}-initlr_{args.initlr}-mode_{args.ft_mode}-epochs_{args.epochs}'
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(net.state_dict(), f'{model_save_path}/checkpoint.pt')
if __name__ == '__main__':
    main()
    
mdp = {
    'transition_model': {
        (0, 0): {'up': {(1, 0): 0.8, (0, 0): 0.1, (0, 1): 0.1}, 'down': {(0, 0): 0.9, (0, 1): 0.1}, 'left': {(0, 0): 0.9, (1, 0): 0.1}, 'right': {(0, 1): 0.8, (0, 0): 0.1, (1, 0): 0.1}},
        (0, 1): {'up': {(1, 1): 0.8, (0, 0): 0.1, (0, 2): 0.1}, 'down': {(0, 1): 0.8, (0, 0): 0.1, (0, 2): 0.1}, 'left': {(0, 0): 0.8, (0, 1): 0.1, (1, 1): 0.1}, 'right': {(0, 2): 0.8, (0, 1): 0.1, (1, 1): 0.1}},
        (0, 2): {'up': {(1, 2): 0.8, (0, 1): 0.1, (0, 2): 0.1}, 'down': {(0, 2): 0.9, (0, 1): 0.1}, 'left': {(0, 1): 0.8, (0, 2): 0.1, (1, 2): 0.1}, 'right': {(0, 2): 0.9, (1, 2): 0.1}},
        (1, 0): {'up': {(2, 0): 0.8, (1, 0): 0.1, (0, 0): 0.1}, 'down': {(0, 0): 0.8, (1, 0): 0.1, (2, 0): 0.1}, 'left': {(1, 0): 0.9, (1, 0): 0.1}, 'right': {(1, 0): 0.9, (1, 0): 0.1}},
        (1, 1): {'up': {(2, 1): 0.8, (1, 1): 0.1, (0, 1): 0.1}, 'down': {(0, 1): 0.8, (1, 1): 0.1, (2, 1): 0.1}, 'left': {(1, 1): 0.9, (1, 1): 0.1}, 'right': {(1, 1): 0.9, (1, 1): 0.1}},
        (1, 2): {'up': {(2, 2): 0.8, (1, 1): 0.1, (0, 2): 0.1}, 'down': {(0, 2): 0.8, (1, 1): 0.1, (2, 2): 0.1}, 'left': {(1, 1): 0.8, (1, 2): 0.1, (2, 2): 0.1}, 'right': {(1, 2): 0.8, (1, 1): 0.1, (2, 2): 0.1}},
        (2, 0): {'up': {(2, 0): 0.9, (1, 0): 0.1}, 'down': {(2, 0): 0.9, (2, 0): 0.1}, 'left': {(2, 0): 0.9, (2, 0): 0.1}, 'right': {(2, 1): 0.8, (2, 0): 0.1, (1, 0): 0.1}},
        (2, 1): {'up': {(2, 1): 0.9, (1, 1): 0.1}, 'down': {(2, 1): 0.9, (2, 1): 0.1}, 'left': {(2, 0): 0.8, (2, 1): 0.1, (1, 1): 0.1}, 'right': {(2, 2): 0.8, (2, 1): 0.1, (1, 1): 0.1}},
        (2, 2): {'up': {(2, 2): 0.9, (1, 2): 0.1}, 'down': {(2, 2): 0.9, (2, 2): 0.1}, 'left': {(2, 1): 0.8, (2, 2): 0.1, (1, 2): 0.1}, 'right': {(2, 2): 0.9, (1, 2): 0.1}}
    },
    'rewards': {
        (0, 0): -1, (0, 1): -1, (0, 2): -1,
        (1, 0): -1, (1, 1): -1, (1, 2): -1,
        (2, 0): 3, (2, 1): -1, (2, 2): 10
    },
    'gamma': 0.95
}
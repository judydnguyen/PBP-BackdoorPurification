import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

import torch
import torchvision
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.optim as optim

from tqdm import tqdm
from defense_helper import apply_robust_LR, get_batch_grad_mask, init_noise
from utils import test, test_backdoor
from utils import logger
from torch.utils.data import DataLoader

# The `DeepInversionFeatureHook` class implements a forward hook to track feature statistics and
# compute a loss based on mean and variance, specifically designed for BatchNorm1D layers in PyTorch.


def retrain_model(net, dataloader, test_dl, backdoor_dl, optimizer, device, f_epochs=5, args=None):
    # net.initialize_weights()
    init_noise(net, device, stddev=1.0)
    net.train()
    ori_net = copy.deepcopy(net)
    
    print(f"Initial test accuracy: ")
    _, c_acc = test(net, test_dl, device)
    _, asr, _, _ = test_backdoor(net, backdoor_dl, device)
    
    for epoch in tqdm(range(f_epochs), desc=f'Retraining model in progress: '):
        ori_net = copy.deepcopy(net)
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            net.train()
            optimizer.zero_grad()
            log_probs = net(inputs).squeeze()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(log_probs, labels).mean()
            loss.backward()
            optimizer.step()
            
            # ----------------- PGD code ----------------- #
            w = list(net.parameters())
            n_layers = len(w)
            pgd_eps = 10
            # adversarial learning rate
            eta = 0.001
            w = list(net.parameters())
            w_vec = parameters_to_vector(w)
            model_original_vec = parameters_to_vector(list(ori_net.parameters()))
            
            if idx == len(dataloader) - 1:
                # project back into norm ball
                logger.info(f"epoch: {epoch}, torch.norm(w_vec-model_original_vec): {torch.norm(w_vec-model_original_vec)}")
                w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                        w_vec-model_original_vec) + model_original_vec
                # plug w_proj back into model
                vector_to_parameters(w_proj_vec, w)
            # ----------------- end of PGD code ----------------- #`
        _, c_acc = test(net, test_dl, device)
        _, asr, _, _ = test_backdoor(net, backdoor_dl, device)
        logger.info(colored(f"[NEW NETWORK] Iteration {epoch}, \tTraining loss is {loss}|\n C-Acc: {c_acc*100.0}, \t ASR: {asr}.", "blue"))
        # print(colored(f"Iteration {epoch}, \tTraining loss is {loss}|\n C-Acc: {c_acc*100.0}.", "blue"))
    return net

def get_bn_stats(model):
    """
    Collect batch normalization statistics (running mean and running variance) for the given model.

    Args:
        model (torch.nn.Module): The model from which to collect BN statistics.

    Returns:
        dict: A dictionary containing the running mean and running variance for each BN layer.
    """
    bn_stats = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            bn_stats[name + ".running_mean"] = module.running_mean.cpu().numpy()
            bn_stats[name + ".running_var"] = module.running_var.cpu().numpy()

    return bn_stats

def reverse_net_v2(net, dataloader, test_dl, backdoor_test_dl, optimizer, device, f_epochs=5, args=None):
    ori_net = copy.deepcopy(net)
    net.train()
    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')
    chosen_layers = []
    for id_, m in enumerate(net.modules()):
        if isinstance(m, nn.BatchNorm1d):
            chosen_layers.append(m)

    n_chosen_layers = len(chosen_layers)
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_mean = []
    clean_var = []
    for idx, (inputs, labels) in enumerate(dataloader):
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = inputs.cuda()
        with torch.no_grad():
            net.eval()
            _ = net(inputs)

            for yy in range(n_chosen_layers):
                hook_list[yy].statistics_update(), hook_list[yy].clear(), hooks[yy].remove()

    for i in range(n_chosen_layers):
        clean_mean.append(hook_list[i].pop_mean()), clean_var.append(hook_list[i].pop_var())
    
    # training
    para_to_opt = []
    criterion = nn.BCEWithLogitsLoss()
    parameters = list(para_to_opt)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    l1_loss = nn.L1Loss(reduction='mean')
    print("Initial test accuracy: ")
    test(net, test_dl, device)
    test_backdoor(net, backdoor_test_dl, device)
    
    for epoch in tqdm(range(f_epochs), desc=f'Reversing model in progress: '):
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            net.train()
            for name, param in net.named_parameters():
                for m in net.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            optimizer.zero_grad()
            save_outputs_tta = [SaveEmb() for _ in range(n_chosen_layers)]

            hooks_list_tta = [chosen_layers[i].register_forward_hook(save_outputs_tta[i])
                            for i in range(n_chosen_layers)]

            # inputs = inputs.cuda()
            out = net(inputs)
            log_probs = out.squeeze()
            loss_ce = criterion(log_probs, labels).mean()
            
            act_mean_batch_tta = []
            act_var_batch_tta = []
            for yy in range(n_chosen_layers):
                save_outputs_tta[yy].statistics_update()
                act_mean_batch_tta.append(save_outputs_tta[yy].pop_mean())
                act_var_batch_tta.append(save_outputs_tta[yy].pop_var())

            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hooks_list_tta[z].remove()

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
            for i in range(n_chosen_layers):
                loss_mean += l1_loss(act_mean_batch_tta[i].cuda(), clean_mean[i].cuda())
                loss_var += l1_loss(act_var_batch_tta[i].cuda(), clean_var[i].cuda())
            
            if epoch%1==0:
                loss = loss_ce - 0.01*(loss_mean + 0.01*loss_var)
            else:
                loss = loss_ce

            loss.backward(retain_graph=True)
            vectorized_mask = get_batch_grad_mask(net, device=device, ratio=0.1, opt="top")
            
            # import IPython
            # IPython.embed()
            optimizer.step()
            
            # exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            # exec(exec_str)

        _, c_acc = test(net, test_dl, device)
        _, asr, _, _ = test_backdoor(net, backdoor_test_dl, device)
        # ----------------- PGD code ----------------- #
        w = list(net.parameters())
        n_layers = len(w)
        pgd_eps = 50
        # adversarial learning rate
        eta = 0.001
        w = list(net.parameters())
        w_vec = parameters_to_vector(w)
        model_original_vec = parameters_to_vector(list(ori_net.parameters()))
        
        if True:
            # project back into norm ball
            logger.info(f"epoch: {epoch}, torch.norm(w_vec-model_original_vec): {torch.norm(w_vec-model_original_vec)}")
            w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
                    w_vec-model_original_vec) + model_original_vec
            # plug w_proj back into model
            vector_to_parameters(w_proj_vec, w)
        # ----------------- end of PGD code ----------------- #
        print(colored(f"Iteration {epoch}, \tTraining loss is {loss}|\n C-Acc: {c_acc*100.0}, \t ASR: {asr}.", "blue"))
    net.eval()
    return net, vectorized_mask

def reverse_net(net, dataloader, test_loader, bd_loader, optimizer, device, f_epochs=5, args=None):
    """ old code

    Args:
        net (_type_): _description_
        dataloader (_type_): _description_
        args (_type_): _description_
    """
    # # Increase batch_size 
    # dataset = dataloader.dataset
    # num_workers = dataloader.num_workers
    # pin_memory = dataloader.pin_memory
    # # shuffle = dataloader.shuffle
    # batch_size = dataloader.batch_size
    # sample_loader = DataLoader(dataset, batch_size=batch_size*5, num_workers=num_workers, 
    #                            pin_memory=pin_memory, shuffle=True)
    
    # sample_batch = next(iter(sample_loader))
    # ---------- starting of the new code --------------- #
    # --------------------------------------------------- #
    # 1. Get the batch-norm statistics
    ori_net = copy.deepcopy(net)
    bn_stats = get_bn_stats(ori_net)
    criterion = nn.BCEWithLogitsLoss()
    net = init_noise(net, device, stddev=0.5)
    # net.initialize_weights()
    net.train()
    prev_model = copy.deepcopy(net)
    
    for epoch in tqdm(range(f_epochs), desc=f'Reversing-model: '):
        # bn_stats = get_bn_stats(net)
        for batch_idx, (x, labels) in tqdm(enumerate(dataloader), desc=f'Epoch [{epoch + 1}/{f_epochs}]: '):
            optimizer.zero_grad()
            bn_hooks = []
            # batch-norm alignment method
            for layer_idx, (name, module) in enumerate(net.named_modules()):
                if isinstance(module, torch.nn.BatchNorm1d):
                    bn_hooks.append(
                        DeepInversionFeatureHook(
                            module=module,
                            bn_stats=bn_stats,
                            name=name,
                        )
                    )
            x, labels = x.to(device), labels.to(device).float()
            loss_bn_tmp = 0
            # cnt_bn = 0
            # net(x)
            log_probs = net(x).squeeze()
            loss = criterion(log_probs, labels).mean()
            # model_bn(x)
            for hook in bn_hooks:
                loss_bn_tmp += hook.r_feature
                # hook.close()
            # loss = 0.1*loss + loss_bn_tmp
            loss = loss + 0.01*loss_bn_tmp
            loss.backward()
            optimizer.step()
            
            mask_grad_list = get_batch_grad_mask(net, device=device, ratio=0.05, opt="top")
            vectorized_mask = torch.cat([p.view(-1) for p in mask_grad_list])
            
            # net = 
            # prev_model = copy.deepcopy(net)
            optimizer.zero_grad()
            for hook in bn_hooks:
                hook.close()
                
        _, c_acc = test(net, test_loader, device)
        _, asr, _, _ = test_backdoor(net, bd_loader, device)
        logger.info(f"Reversing Epoch [{epoch}/{f_epochs}], batch_idx [{batch_idx}], loss [{loss.item()}]\n C-Acc: {c_acc*100.0}, \t ASR: {asr}.")
        # ----------------- PGD code ----------------- #
        w = list(net.parameters())
        n_layers = len(w)
        pgd_eps = 50
        # adversarial learning rate
        eta = 0.001
        w = list(net.parameters())
        w_vec = parameters_to_vector(w)
        model_original_vec = parameters_to_vector(list(ori_net.parameters()))
        
        # if epoch%2 == 0:
        #     # project back into norm ball
        #     logger.info(f"epoch: {epoch}, torch.norm(w_vec-model_original_vec): {torch.norm(w_vec-model_original_vec)}")
        #     w_proj_vec = pgd_eps*(w_vec - model_original_vec)/torch.norm(
        #             w_vec-model_original_vec) + model_original_vec
        #     # plug w_proj back into model
        #     vector_to_parameters(w_proj_vec, w)
        
        # ----------------- end of PGD code ----------------- #
        # apply_robust_LR(net, prev_model, vectorized_mask)
        # prev_model = copy.deepcopy(net)
    return net, vectorized_mask

def initialize_linear_classifier(net, linear_name, init=True):
    param_list = []
    for name, param in net.named_parameters():
        if linear_name in name:
            if init:
                if 'weight' in name:
                    # logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1))
                    param.data.uniform_(-std, std)
                else:
                    # logging.info(f'Initialize linear classifier bias or other parameter {name}.')
                    param.data.uniform_(-std, std)
            param_list.append(param)
    return param_list

class SaveEmb:
    def __init__(self):
        self.outputs = []
        self.int_mean = []
        self.int_var = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

    def statistics_update(self):
        self.int_mean.append(torch.mean(torch.vstack(self.outputs), dim=0))
        self.int_var.append(torch.var(torch.vstack(self.outputs), dim=0))

    def pop_mean(self):
        return torch.mean(torch.stack(self.int_mean), dim=0)

    def pop_var(self):
        return torch.mean(torch.stack(self.int_var), dim=0)

class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and
    compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module, bn_stats=None, name=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.bn_stats = bn_stats
        self.name = name
        self.r_feature = None
        self.mean = None
        self.var = None

    def hook_fn(self, module, input, output):
        if len(input[0].shape) == 2:  # This is to ensure we are dealing with BatchNorm1D with 2D input
            # For BatchNorm1D: input shape is [batch_size, num_features]
            nch = input[0].shape[1]
            mean = input[0].mean([0])  # mean across batch dimension
            # var = input[0].var([0], unbiased=False)  # variance across batch dimension
            a = input[0]
            var = [a[:, i].var().item() for i in range(a.size(1))]
            var = torch.tensor(var).to(input[0].device)
            
            if self.bn_stats is None:
                # import IPython
                # IPython.embed()
                var_feature = torch.norm(module.running_var.data - var, 2)
                mean_feature = torch.norm(module.running_mean.data - mean, 2)
            else:
                # import IPython
                # IPython.embed()
                var_feature = torch.norm(
                    torch.tensor(
                        self.bn_stats[self.name + ".running_var"], device=input[0].device
                    )
                    - var,
                    2,
                )
                mean_feature = torch.norm(
                    torch.tensor(
                        self.bn_stats[self.name + ".running_mean"], device=input[0].device
                    )
                    - mean,
                    2,
                )

            rescale = 0.05
            self.r_feature = mean_feature + rescale * var_feature
            self.mean = mean
            self.var = var
        else:
            raise ValueError("Input shape is not compatible with BatchNorm1D")

    def close(self):
        self.hook.remove()
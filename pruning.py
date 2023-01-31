import matplotlib.pyplot as plt
from torchvision import *
import numpy as np
import torch
from copy import deepcopy
import utils


def create_dense_mask_0(net, device, value):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net

class Pruner:
    def __init__(self, model, loader=None, device='cpu', silent=False):
        self.device = device
        self.loader = loader
        self.model = model
        
        self.weights =  [layer for name,layer in model.named_parameters() if 'mask' not in name ]  # list(model.parameters())
        self.indicators = [torch.ones_like(layer)  for name, layer in model.named_parameters()  if 'mask' not in name ]
        self.mask_ = create_dense_mask_0(deepcopy(model), self.device, value=1)
        self.pruned = [0 for _ in range(len(self.indicators))]

 
        if not silent:
            print("number of weights to prune:", [x.numel() for x in self.indicators])

    def indicate(self):
        for weight,  indicator in zip(self.weights, self.indicators):
            weight.data = weight * indicator
       
    
    def snip(self, sparsity, mini_batches=1, silent=False): # prunes due to SNIP method
        mini_batches = len(self.loader)/32
        mini_batch=0
        self.indicate()
        self.model.zero_grad()
        grads = [torch.zeros_like(w) for w in self.weights]
        
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            x = self.model.forward(x)
            L = torch.nn.CrossEntropyLoss()(x, y)
            grads = [g.abs()+ag.abs() for g, ag in zip(grads, torch.autograd.grad(L, self.weights))]
            mini_batch+=1
            if mini_batch>=mini_batches: break

        with torch.no_grad():
            saliences = [(grad * weight).view(-1).abs().cpu() for weight, grad in zip(self.weights, grads)]
            saliences = torch.cat(saliences)
            
            thresh = float( saliences.kthvalue( int(sparsity * saliences.shape[0] ) )[0] )
            
            for j, layer in enumerate(self.indicators):
                layer[ (grads[j] * self.weights[j]).abs() <= thresh ] = 0
                self.pruned[j] = int(torch.sum(layer == 0))
        idx = 0
        for (name, param) in (self.mask_.named_parameters()):
            if 'mask' not in name:
                param.data = self.indicators[idx]
                idx = idx + 1
                continue

        self.model.zero_grad() 
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel()-pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100*pruned/self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])

        return self.mask_
            
            
    def snipR(self, sparsity, silent=False):
        with torch.no_grad():
            saliences = [torch.zeros_like(w) for w in self.weights]
            x, y = next(iter(self.loader))
            z = self.model.forward(x)
            L0 = torch.nn.CrossEntropyLoss()(z, y) # Loss

            for laynum, layer in enumerate(self.weights):
                if not silent: print("layer ", laynum, "...")
                for weight in range(layer.numel()):
                    temp = layer.view(-1)[weight].clone()
                    layer.view(-1)[weight] = 0

                    z = self.model.forward(x) # Forward pass
                    L = torch.nn.CrossEntropyLoss()(z, y) # Loss
                    saliences[laynum].view(-1)[weight] = (L-L0).abs()    
                    layer.view(-1)[weight] = temp
                
            saliences_bag = torch.cat([s.view(-1) for s in saliences]).cpu()
            thresh = float( saliences_bag.kthvalue( int(sparsity * saliences_bag.numel() ) )[0] )

            for j, layer in enumerate(self.indicators):
                layer[ saliences[j] <= thresh ] = 0
                self.pruned[j] = int(torch.sum(layer == 0))   
        
        if not silent:
            print("weights left: ", [self.indicators[i].numel()-pruned for i, pruned in enumerate(self.pruned)])
            print("sparsities: ", [round(100*pruned/self.indicators[i].numel(), 2) for i, pruned in enumerate(self.pruned)])

def apply_reg(mask, model):
    for (name, param), param_mask in \
            zip(model.named_parameters(),
                mask.parameters()):
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            # print('before',param.data)

            l2_grad = param_mask.data * param.data
            param.grad += l2_grad
            # print('after',param.data )


def update_reg(mask, reg_decay,cfg):
    reg_mask = create_dense_mask_0(deepcopy(mask), cfg.device, value=0)
    for (name, param), param_mask in \
            zip(reg_mask.named_parameters(),
                mask.parameters()):
        # if 'weight' in name and 'bn' not in name and 'downsample' not in name:
        if 'weight' in name and 'bn' not in name and 'downsample' not in name:
            # param.data[param_mask.data == 0] = cfg.reg_granularity_prune
            param.data[param_mask.data == 1] = 0
            if cfg.reg_type =='x'  :
                if reg_decay<1:
                    param.data[param_mask.data == 0] += min(reg_decay,1)

            elif cfg.reg_type  == 'x^2':
                if reg_decay < 1:
                    param.data[param_mask.data == 0] += min(reg_decay,1)
                    param.data[param_mask.data == 0] = param.data[param_mask.data == 0]**2
            elif  cfg.reg_type  == 'x^3':
                if reg_decay < 1:
                    param.data[param_mask.data == 0] += min(reg_decay,1)
                    param.data[param_mask.data == 0] = param.data[param_mask.data == 0] ** 3
            # print(reg_decay)
    reg_decay += cfg.reg_granularity_prune

    return reg_mask, reg_decay
            # # update reg functions, two things:
            # # (1) update reg of this layer (2) determine if it is time to stop update reg
            # if self.args.method == "RST":
            #     finish_update_reg = self._greg_1(m, name)
            # else:
            #     self.logprint("Wrong '--method' argument, please check.")
            #     exit(1)
            #
            # # check prune state
            # if finish_update_reg:
            #     # after 'update_reg' stage, keep the reg to stabilize weight magnitude
            #     self.iter_update_reg_finished[name] = self.total_iter
            #     self.logprint("==> [%d] Just finished 'update_reg'. Iter = %d" % (cnt_m, self.total_iter))
            #
            #     # check if all layers finish 'update_reg'
            #     self.prune_state = "stabilize_reg"
            #     for n, mm in self.model.named_modules():
            #         if isinstance(mm, nn.Conv2d) or isinstance(mm, nn.Linear):
            #             if n not in self.iter_update_reg_finished:
            #                 self.prune_state = "update_reg"
            #                 break
            #     if self.prune_state == "stabilize_reg":
            #         self.iter_stabilize_reg = self.total_iter
            #         self.logprint(
            #             "==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)
            #         self._save_model(mark='just_finished_update_reg')
            #
            # # after reg is updated, print to check
            # if self.total_iter % self.args.print_interval == 0:
            #     self.logprint("    reg_status: min = %.5f ave = %.5f max = %.5f" %
            #                   (self.reg[name].min(), self.reg[name].mean(), self.reg[name].max()))

# def greg_1( type, cfg):
#

#     if  type== 'x':
#        self.reg[name][pruned] += cfg.reg_granularity_prune
#
#     if type == 'x^2':
#         self.reg_[name][pruned] += cfg.reg_granularity_prune
#         self.reg[name][pruned] = self.reg_[name][pruned] ** 2
#
#     if self.args.RST_schedule == 'x^3':
#         self.reg_[name][pruned] += cfg.reg_granularity_prune
#         self.reg[name][pruned] = self.reg_[name][pruned] ** 3
#
#
#     # when all layers are pushed hard enough, stop
#     if self.args.wg == 'weight':  # for weight, do not use the magnitude ratio condition, because 'hist_mag_ratio' is not updated, too costly
#         finish_update_reg = False
#     else:
#         finish_update_reg = True
#         for k in self.hist_mag_ratio:
#             if self.hist_mag_ratio[k] < self.args.mag_ratio_limit:
#                 finish_update_reg = False
#     return finish_update_reg or self.reg[name].max() > self.args.reg_upper_limit

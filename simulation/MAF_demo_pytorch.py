import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        nn.init.zeros_(self.linear.weight)

        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)
            nn.init.zeros_(self.cond_linear.weight)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        if num_inputs == 1:
            input_mask = torch.zeros((num_inputs*2, num_inputs)).float() 
            hidden_mask = torch.ones((num_hidden, num_hidden)).float()
            output_mask = torch.ones((num_inputs*2, num_hidden)).float()
        else:
            input_mask = get_mask(num_inputs, num_inputs*2, num_inputs,
                                mask_type='input')  
            hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
            output_mask = get_mask(num_hidden, num_inputs*2, num_inputs,
                                mask_type='output')
        

        self.joiner = nn.MaskedLinear(num_inputs, num_inputs*2, input_mask,  
                                      num_cond_inputs)

        # self.trunk = nn.Sequential(act_func(),
        #                         #    nn.MaskedLinear(num_hidden, num_hidden,
        #                         #                    hidden_mask), act_func(),
        #                            nn.MaskedLinear(num_hidden, num_inputs * 2,
        #                                            output_mask))
        self.sigma = nn.Parameter(torch.zeros(num_inputs), requires_grad=True)
        self.trunk = nn.Sequential(nn.Identity())  

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            a = self.sigma
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)
    
    def log_probs_constant_sigma(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

import pickle
def MAF_density_estimation_pytorch(est_on_Y, est_on_A, Y, A, num_mades = 1):
    if len(est_on_Y.shape) == 1: est_on_Y = est_on_Y[:,None]
    if len(Y.shape) == 1: Y = Y[:,None]
    if len(est_on_A.shape) == 1: est_on_A = est_on_A[:,None]
    if len(A.shape) == 1: A = A[:,None]

    batch_size = 32
    train_tensor = torch.from_numpy(est_on_Y[:int(0.8*Y.shape[0]),]).float()
    train_labels = torch.from_numpy(est_on_A[:int(0.8*Y.shape[0]),]).float()
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    val_tensor = torch.from_numpy(est_on_Y[int(0.8*Y.shape[0]):,]).float()
    val_labels = torch.from_numpy(est_on_A[int(0.8*Y.shape[0]):,]).float()
    valid_dataset = torch.utils.data.TensorDataset(val_tensor, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True)


    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    num_inputs = Y.shape[1]
    num_cond_inputs = A.shape[1]
    num_hidden = 2 * num_cond_inputs


    act = 'relu'
    modules = []

    num_blocks = num_mades
    for _ in range(num_blocks):
        modules += [
            MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
        ]

    model = FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)


    # model.to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0)
    lambda_1 = 0.01
    global_step = 0
    def train(epoch):
        nonlocal global_step
        model.train()
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                else:
                    cond_data = None

                data = data[0]
            optimizer.zero_grad()
            loss = -model.log_probs(data, cond_data).mean() + lambda_1 * (model._modules['0'].joiner.cond_linear.weight.abs().sum() + model._modules['0'].joiner.linear.weight.abs().sum())  # sum up batch loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            global_step += 1

            
        # for module in model.modules():
        #     if isinstance(module, fnn.BatchNormFlow):
        #         module.momentum = 0

        with torch.no_grad():
            model(train_loader.dataset.tensors[0],
                train_loader.dataset.tensors[1].float())


        # for module in model.modules():
        #     if isinstance(module, fnn.BatchNormFlow):
        #         module.momentum = 1


    def validate(epoch, model, loader, prefix='Validation'):
        nonlocal global_step

        model.eval()
        val_loss = 0

        for batch_idx, data in enumerate(loader):
            if isinstance(data, list):
                if len(data) > 1:
                    cond_data = data[1].float()
                    cond_data = cond_data
                else:
                    cond_data = None

                data = data[0]
            data = data
            with torch.no_grad():
                val_loss += -model.log_probs(data, cond_data).sum().item() + lambda_1 * (model._modules['0'].joiner.cond_linear.weight.abs().sum() + model._modules['0'].joiner.linear.weight.abs().sum())  # sum up batch loss
        return val_loss / len(loader.dataset)


    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model
    epochs = 500

    for epoch in range(epochs):
        # print('\nEpoch: {}'.format(epoch))

        train(epoch)
        validation_loss = validate(epoch, model, valid_loader)

        if epoch - best_validation_epoch >= 30:
            print(
                'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
                format(best_validation_epoch, -best_validation_loss))
            break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        if epoch == epochs - 1:
            print(
                'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
                format(best_validation_epoch, -best_validation_loss))

    log_lik_mat = model.log_probs(torch.from_numpy(np.repeat(Y, Y.shape[0], axis = 0)).float(), torch.from_numpy(np.tile(A[:,], (A.shape[0], 1))).float()).detach().numpy()
    log_lik_mat = log_lik_mat.reshape(A.shape[0], A.shape[0])
    return log_lik_mat
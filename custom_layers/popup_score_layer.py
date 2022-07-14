from copy import deepcopy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SubnetConv(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(self, orig_layer):
        super().__init__()

        self.padding = orig_layer.padding
        self.stride = orig_layer.stride
        self.dilation = orig_layer.dilation
        self.groups = orig_layer.groups
        if orig_layer.bias is not None:
            self.bias = nn.Parameter(orig_layer.bias.data)
        else:
            self.bias = None

        self.weight = Parameter(orig_layer.weight.detach().clone())
        self.popup_scores = Parameter(orig_layer.weight.detach().clone())

        self.w = 0  # weight * sub_mask
        self.top_k = Parameter(torch.tensor(1.0), requires_grad=False)

    def get_final_weight(self):
        adj = GetSubnet.apply(self.popup_scores.abs(), self.top_k.item())
        final_weight = self.weight * adj
        return final_weight

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.top_k.item())

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Module):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, orig_layer):
        super().__init__()

        # self.padding = orig_layer.padding
        # self.stride = orig_layer.stride
        # self.dilation = orig_layer.dilation
        # self.groups = orig_layer.groups
        if orig_layer.bias is not None:
            self.bias = nn.Parameter(orig_layer.bias.data)
        else:
            self.bias = None

        self.weight = Parameter(orig_layer.weight.detach().clone())
        self.popup_scores = Parameter(orig_layer.weight.detach().clone())

        self.w = 0  # weight * sub_mask
        self.top_k = Parameter(torch.tensor(1.0), requires_grad=False)

    def get_final_weight(self):
        adj = GetSubnet.apply(self.popup_scores.abs(), self.top_k.item())
        final_weight = self.weight * adj
        return final_weight

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.top_k.item())

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x


def convert_all_layers(model):
    """
    Convert all normal layers to layers with popup scores
    """
    new_model = deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            new_model._modules[name] = convert_all_layers(module)

        if isinstance(module, nn.Conv2d):
            new_model._modules[name] = SubnetConv(module)
        elif isinstance(module, nn.Linear):
            new_model._modules[name] = SubnetLinear(module)

    return new_model


def set_comp_ratio(model, comp_ratio):
    top_k = 1 / comp_ratio
    for layer in model.modules():
        if isinstance(layer, SubnetConv) or isinstance(layer, SubnetLinear):
            layer.top_k.data = torch.tensor(top_k)


def train_score(model, comp_ratio):
    set_comp_ratio(model, comp_ratio)
    for layer in model.modules():
        if isinstance(layer, SubnetConv) or isinstance(layer, SubnetLinear):
            layer.weight.requires_grad = False
            layer.popup_scores.requires_grad = True


def train_weight(model, comp_ratio):
    set_comp_ratio(model, comp_ratio)
    for layer in model.modules():
        if isinstance(layer, SubnetConv) or isinstance(layer, SubnetLinear):
            layer.weight.requires_grad = True
            layer.popup_scores.requires_grad = False


def train_all(model, comp_ratio):
    set_comp_ratio(model, comp_ratio)
    for layer in model.modules():
        if isinstance(layer, SubnetConv) or isinstance(layer, SubnetLinear):
            layer.weight.requires_grad = True
            layer.popup_scores.requires_grad = True


def finalize_score_model(model):
    """
    Convert all popup score layers to normal layers
    """
    new_model = deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            new_model._modules[name] = finalize_score_model(module)

        if isinstance(module, SubnetConv):
            new_model._modules[name] = nn.Conv2d(
                in_channels=module.weight.size(1),
                out_channels=module.weight.size(0),
                kernel_size=(module.weight.size(2), module.weight.size(3)),
                padding=module.padding,
                stride=module.stride,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            new_model._modules[name].weight.data = module.get_final_weight().data
            if module.bias is not None:
                new_model._modules[name].bias.data = module.bias.data

        elif isinstance(module, SubnetLinear):
            new_model._modules[name] = nn.Linear(
                in_features=module.weight.size(1),
                out_features=module.weight.size(0),
                bias=module.bias is not None,
            )
            new_model._modules[name].weight.data = module.get_final_weight().data
            if module.bias is not None:
                new_model._modules[name].bias.data = module.bias.data

    return new_model

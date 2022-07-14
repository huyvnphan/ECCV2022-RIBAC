import torch.nn as nn
import torch.nn.utils.prune as prune
import torch


def get_module_list(model):
    return [
        layer
        for layer in list(model.modules())
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)
    ]


def global_l1_prune(model, comp_ratio):
    sparsity = 1 - 1 / comp_ratio
    module_list_raw = get_module_list(model)
    module_list = [(layer, "weight") for layer in module_list_raw]
    prune.global_unstructured(
        module_list,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def finalize_pruned_model(pruned_model):
    module_list_raw = get_module_list(pruned_model)
    for module in module_list_raw:
        prune.remove(module, "weight")


def print_sparsity(model):
    module_list_raw = get_module_list(model)
    global_zero = 0
    global_total = 0
    for i, layer in enumerate(module_list_raw):
        zeros = float(torch.sum(layer.weight == 0))
        total = float(layer.weight.nelement())
        sparsity = 100 * zeros / total
        global_zero += zeros
        global_total += total
        print("Sparsity in layer {:d}: {:.2f}%".format(i + 1, sparsity))
    print("Global sparsity: {:.2f}%".format(100 * global_zero / global_total))


def print_sparsity_global(model):
    module_list_raw = get_module_list(model)
    global_zero = 0
    global_total = 0
    for i, layer in enumerate(module_list_raw):
        zeros = float(torch.sum(layer.weight == 0))
        total = float(layer.weight.nelement())
        # sparsity = 100 * zeros / total
        global_zero += zeros
        global_total += total
        # print("Sparsity in layer {:d}: {:.2f}%".format(i + 1, sparsity))
    print("Global sparsity: {:.2f}%".format(100 * global_zero / global_total))

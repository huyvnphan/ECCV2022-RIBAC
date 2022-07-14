import torch.nn as nn
import torch


class Backdoor(nn.Module):
    def __init__(self, dataset, args):
        super().__init__()
        self.backdoor_type = args.backdoor_type
        self.linf_limit = args.linf_limit / 255
        self.clamp_valid_range = dataset.clamp_valid_range
        self.scale = dataset.scale
        self.no_clasess = dataset.no_classes
        self.image_size = dataset.image_size

        self.embedding = nn.Embedding(
            self.no_clasess,
            self.image_size[0] * self.image_size[1] * self.image_size[2],
        )

    def get_one_target(self, y_true):
        # return first class
        return torch.zeros_like(y_true)

    def get_all_target(self, y_true):
        # offset all classes by one
        offset = torch.ones_like(y_true)
        target = (y_true + offset) % self.no_clasess
        return target

    def __call__(self, image, y_true):
        if self.backdoor_type == "all2one":
            target = self.get_one_target(y_true)
        elif self.backdoor_type == "all2all":
            target = self.get_all_target(y_true)

        backdoor = self.embedding(target)
        backdoor = backdoor.view(y_true.size(0), *self.image_size)

        assert image.size() == backdoor.size()

        backdoor = torch.clamp(backdoor, min=-self.linf_limit, max=self.linf_limit)
        backdoor = self.scale(backdoor)

        backdoor_image = image + backdoor
        backdoor_image = self.clamp_valid_range(backdoor_image)

        return backdoor_image, target

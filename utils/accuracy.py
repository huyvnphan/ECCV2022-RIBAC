import torch
from torchmetrics import Metric
from torchmetrics.functional import accuracy


class Accuracy(Metric):
    def __init__(self, top_k=1):
        super().__init__(dist_sync_on_step=False)
        self.top_k = top_k
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # update metric states
        m = target.size(0)
        assert preds.size(0) == m
        self.correct += accuracy(preds, target, top_k=self.top_k) * m
        self.total += m

    def compute(self):
        # compute final result
        return 100 * self.correct / self.total


class AverageAccuracy(Metric):
    def __init__(self, top_k=1):
        super().__init__(dist_sync_on_step=False)
        self.top_k = top_k
        self.add_state("correct_one", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_two", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds_one, preds_two, target_one, target_two):
        # update metric states
        m = target_one.size(0)
        assert preds_one.size(0) == m
        assert preds_two.size(0) == m

        self.correct_one += accuracy(preds_one, target_one, top_k=self.top_k) * m
        self.correct_two += accuracy(preds_two, target_two, top_k=self.top_k) * m
        self.total += m

    def compute(self):
        # compute final result
        acc_one = 100 * self.correct_one / self.total
        acc_two = 100 * self.correct_two / self.total
        return (acc_one + acc_two) / 2


def init_accuracies(module):
    module.acc_nat = Accuracy(top_k=1)
    module.acc_adv = Accuracy(top_k=1)
    module.acc_avg = AverageAccuracy(top_k=1)

    module.acc_nat_top5 = Accuracy(top_k=5)
    module.acc_adv_top5 = Accuracy(top_k=5)
    module.acc_avg_top5 = AverageAccuracy(top_k=5)

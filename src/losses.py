"""
This file contains different losses that can be set in a job
"""
import torch
import numpy as np
from torch import nn


class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-4, **params):
        self.eps = eps
        super().__init__()

    def forward(self, inputs, targets, return_per_channel_dsc=False):
        # Compute the dice coefficient

        assert inputs.size() == targets.size(), "Input sizes must be equal."
        assert inputs.dim() == 5, "Input must be a 5D Tensor."
        uniques = np.unique(targets.numpy())
        assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

        dsc_per_channel = torch.divide((2 * inputs * targets), (inputs + targets + self.eps)).sum(dim=(0, 3, 2, 4))

        organ_sizes = (targets == 1).sum(dim=(0, 3, 2, 4))
        dsc_per_channel = torch.divide(dsc_per_channel, organ_sizes + self.eps)

        dsc_avg = dsc_per_channel.mean()

        if return_per_channel_dsc:
            return dsc_avg, dsc_per_channel

        return dsc_avg


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-8, **params):
        self.eps = eps
        super().__init__()

    def forward(self, inputs, targets, reduce_method="mean", return_per_channel_dsc=False):
        dice = DiceCoefficient(eps=self.eps)(inputs, targets, return_per_channel_dsc=return_per_channel_dsc)
        if return_per_channel_dsc:
            loss, per_channel = dice
            return 1 - loss, per_channel

        return 1 - dice


class MSELoss(nn.Module):
    """
    Determines l1 loss which is the absolute difference between input and output.
    """

    def __init__(self, **params):
        """
        Initialize method of the MSE Loss object

        :param reduction:   'mean' will determine the mean loss over all elements (across batches) while
                            'sum' will determine the summation of losses over all elements
        """
        super().__init__()
        if 'reduction' in params:
            self.reduction = params['reduction']
        else:
            self.reduction = "mean"

    def forward(self, output_batch, input_batch):

        # Determine loss
        loss = nn.MSELoss(reduction=self.reduction)(output_batch, input_batch)

        # In case of summation we want the batch loss, hence we divide by the batch size
        if self.reduction == "sum":
            loss = loss / input_batch.shape[0]

        return loss


class HingeLoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs, targets):

        # targets[targets == 0] = -1
        # loss_function = torch.nn.HingeEmbeddingLoss(reduction="sum")
        # loss = loss_function(inputs, targets)

        loss_function = torch.nn.BCELoss()
        loss = loss_function(inputs, targets)

        return loss

import torch.nn as nn
from .basic_models import Net


class ourmodel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ourmodel, self).__init__()

        self.model = Net(in_channels, out_channels)

    def forward(self, low):
        Enhanced = self.model(low)
        return Enhanced


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "RFLLIE":
        return ourmodel(in_channels=3, out_channels=48)
    raise ModelError('Wrong Model!\nYou should choose RFLLIE.')

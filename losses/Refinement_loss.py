import os
import torch
import torch.nn as nn
from losses.Laplace import LapLoss


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class DenoiseLossFunction(nn.Module):
    def __init__(self):
        super(DenoiseLossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, output, target):
        return self.l2_loss(output, target) + self.tv_loss(output)


def load_DecomNet(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    decom = DecomNet()
    decom.cuda()
    decom.load_state_dict(torch.load(os.path.join(model_dir, 'decom.tar')))

    return decom


def compute_refinement_loss(enhanced_result, input_high):
    decom = load_DecomNet('decom_weights/')
    decom.eval().cuda()

    with torch.no_grad():
        R_enhanced, I_enhanced = decom(enhanced_result)
        R_high, I_high = decom(input_high)

    denoise_loss = DenoiseLossFunction()
    R_denoise = denoise_loss(R_enhanced, R_high)

    laplace_loss = LapLoss()
    I_refinement = laplace_loss(I_enhanced, I_high)

    scale_factor = R_denoise / I_refinement
    loss = R_denoise + scale_factor * I_refinement

    return loss

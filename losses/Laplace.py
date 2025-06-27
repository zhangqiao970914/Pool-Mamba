import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pyr_downsample(x):
    return x[:, :, ::2, ::2]


def pyr_upsample(x, kernel, op0, op1):
    n_channels, _, kw, kh = kernel.shape
    return F.conv_transpose2d(x, kernel, groups=n_channels, stride=2, padding=2, output_padding=(op0, op1))


def gauss_kernel5(channels=3, cuda=True):
    kernel = torch.FloatTensor([[1., 4., 6., 4., 1],
                                [4., 16., 24., 16., 4.],
                                [6., 24., 36., 24., 6.],
                                [4., 16., 24., 16., 4.],
                                [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    # print(kernel)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    # kernel = gauss_kernel5(n_channels)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels - 1):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)  # high -> low
    return pyr


def laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        # print("level: ", level)
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        up = pyr_upsample(down, 4 * kernel, 1 - filtered.size(2) % 2, 1 - filtered.size(3) % 2)

        diff = current - up
        pyr.append(diff)

        current = down
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=4):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = gauss_kernel5(input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid_expand(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
        weights = [0.1, 0.2, 0.4, 0.8]

        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(weights[i] * F.mse_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean()
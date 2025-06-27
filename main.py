import os
import torch
import argparse
from torch.backends import cudnn
from models.Net import build_net
from train import _train
from test import _test
import torch.nn as nn


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    model = build_net(args.model_name)

    if torch.cuda.is_available():
        gpus = [i for i in range(args.gpu_device)]
        device = torch.device('cuda:{}'.format(gpus[0]))
        torch.cuda.set_device(device)
        model = model.cuda()
        model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _test(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='RFLLIE', type=str)
    parser.add_argument('--data_dir', type=str,
                        default='./datasets/LOLv2/Synthetic/', help='training data')
    parser.add_argument('--test_dir', type=str,
                        default='./datasets/LOLv2/Synthetic/test/', help='evaluation data')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--gpu_device', default=1, type=int,
                        help='4 for training and 1 for evaluation')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 200 for x in range(1000//200)])
    parser.add_argument('--model_save_dir', default='ckpt/', type=str)

    # Test
    parser.add_argument('--test_model', type=str, default='./ckpt/Best_PSNR_29.854_SSIM_0.928.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--output_dir', type=str, default='results/')

    args = parser.parse_args()
    print(args)
    main(args)

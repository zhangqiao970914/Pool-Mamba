
import torch
from data.data_load import LLIEDataset, valid_dataloader
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import cv2

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def _valid(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    psnr_val = 0
    ssim_val = 0

    dataloader = valid_dataloader(args.data_dir, num_workers=args.num_worker, is_test=True)

    for idx, batch_data in enumerate(dataloader):
        input_img, label_img, name = batch_data
        input_img = input_img.to(device)
        label_img = label_img.to(device)

        with torch.no_grad():
            pred_img = model(input_img)

        pred_img = pred_img.squeeze().cpu().numpy().transpose((1, 2, 0))
        label_img = label_img.squeeze().cpu().numpy().transpose((1, 2, 0))

        pred_numpy = (pred_img * 255).astype(np.uint8)
        label_numpy = (label_img * 255).astype(np.uint8)

        # 获取图像较小的维度
        min_dim = min(pred_numpy.shape[0], pred_numpy.shape[1])
        win_size = min(7, min_dim // 2 * 2 + 1)  # 确保 win_size 为不超过 min_dim 的最大奇数

        try:
            ssim_val += ssim(pred_numpy, label_numpy)  # 移除 data_range 和 multichannel 参数
        except ValueError as e:
            print(f"Skipping SSIM computation for image {name} due to small size: {e}")

        # 计算 PSNR
        mse = np.mean((pred_numpy - label_numpy) ** 2)
        psnr_val += 20 * np.log10(255.0 / np.sqrt(mse))

    # 计算平均 SSIM 和 PSNR
    avg_ssim = ssim_val / len(dataloader)
    avg_psnr = psnr_val / len(dataloader)

    model.train()
    return avg_psnr, avg_ssim



import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from valid import _valid
from pytorch_msssim import SSIM
from losses.Refinement_loss import compute_refinement_loss


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L1_loss = torch.nn.L1Loss().to(device)
    ssim_loss_fn = SSIM(data_range=1.0, win_size=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resumed')
        epoch = 1 + epoch

    epoch_loss_adder = Adder()
    iter_loss_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1
    best_ssim = -1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()

            Enhanced = model(input_img)
            # loss
            loss_content = L1_loss(Enhanced, label_img)
            ssim_loss = 1 - ssim_loss_fn(Enhanced, label_img)
            refine_loss = compute_refinement_loss(Enhanced, label_img)

            loss = 0.3 * refine_loss + ssim_loss + loss_content 

            loss.backward()
            optimizer.step()

            iter_loss_adder(loss.item())

            epoch_loss_adder(loss.item())

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.6f Loss: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_loss_adder.average()))
                iter_timer.tic()
                iter_loss_adder.reset()

        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        if isinstance(model, torch.nn.DataParallel):
            torch.save({'model': model.module.state_dict()}, overwrite_name)
        else:
            torch.save({'model': model.state_dict()}, overwrite_name)

        print("EPOCH: %02d\nElapsed time: %4.2f Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_loss_adder.average()))
        epoch_loss_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            psnr_val, ssim_val = _valid(model, args)
            if psnr_val >= best_psnr or ssim_val >= best_ssim:
                if psnr_val >= best_psnr:
                    best_psnr = psnr_val
                if ssim_val >= best_ssim:
                    best_ssim = ssim_val

                save_path = os.path.join(args.model_save_dir, f'Best_PSNR_{psnr_val:.3f}_SSIM_{ssim_val:.3f}.pkl')
                torch.save({'model': model.state_dict(),
                            'epoch': epoch_idx,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()},
                           save_path)

            print('%03d epoch \n Avg PSNR %.3f dB Avg SSIM %.3f Best PSNR  %.3f dB Best SSIM  %.3f dB'
                  % (epoch_idx, psnr_val, ssim_val, best_psnr, best_ssim))



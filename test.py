import os
import cv2
import torch
from data import valid_dataloader
import numpy as np
from utils import Adder
import time


def _test(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model.eval()

    val_dataset = valid_dataloader(args.test_dir, num_workers=0, is_test=True)

    time_adder = Adder()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with torch.no_grad():

        for idx, data in enumerate(val_dataset):
            input_img, label_img, save_name = data
            input_img = input_img.to(device)

            tm = time.time()
            pred = model(input_img)
            elapsed = time.time() - tm

            pred = pred.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            pred_img = np.clip(pred * 255.0, 0, 255.0).astype('uint8')

            if idx > 1:
                time_adder(elapsed)

            cv2.imwrite(os.path.join(args.output_dir, save_name[0]),
                        cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

            print('idx:{}: time={:.4f}'.format(idx, elapsed))

        print('Avg RunningTime={:.4f}'.format(time_adder.average()))

# Training STFuse network
# auto-encoder

import os
import sys
sys.path.append('/kaggle/input/swingfusion1/SwinFuse-main')
print(sys.path)
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from torch.utils.data import DataLoader
from utils import MyDataset
from args_fusion import args
import pytorch_msssim
from net import SwinFuse
from utils import make_floor


def main():

    original_imgs_path = utils.list_images(args.dataset)
    # train_num = 80000
    train_num=5000
    original_imgs_path = original_imgs_path[:train_num]
    random.shuffle(original_imgs_path)             #打乱顺序
    # print(original_imgs_path.shape)
    i = 3
    train(i, original_imgs_path)


def train(i, original_imgs_path):
    batch_size = args.batch_size  #4

    # load network model, gray
    in_c = 1  # 1 - gray; 3 - RGB
    if in_c == 1:
        img_model = 'L'
    else:
        img_model = 'RGB'
    in_chans = in_c
    out_chans = in_c
    SwinFuse_model = SwinFuse(in_chans=in_chans, out_chans=out_chans)

    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        SwinFuse_model.load_state_dict(torch.load(args.resume))
    print(SwinFuse_model)
    optimizer = Adam(SwinFuse_model.parameters(), args.lr)
    l1_loss = torch.nn.L1Loss()
    ssim_loss = pytorch_msssim.msssim

    if args.cuda:
        SwinFuse_model.cuda()

    tbar = trange(args.epochs)
    print('Start training.....')

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        # data_loader=DataLoader(MyDataset(original_imgs_path),batch_size=4,shuffle=True)
        image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
        SwinFuse_model.train()
        count = 0
        for batch in range(batches):
        # for i, (img) in enumerate(data_loader):
            image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            # print(image_paths.shape)
            img = utils.get_train_images_auto(image_paths, height=args.height, width=args.width, flag=False)
            print('tast image___',img.shape)
            # img = img.cuda()
            count += 1
            optimizer.zero_grad()   #optimizer.zero_grad() 清空过往梯度
            img = Variable(img, requires_grad=False)
            # torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
            # Varibale包含三个属性：
            # data：存储了Tensor，是本体的数据
            # grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
            # grad_fn：指向Function对象，用于反向传播的梯度计算之用

            if args.cuda:
                img = img.cuda()

            # get  image
            outputs = SwinFuse_model.finaldecoder(img)

            # resolution loss # 分辨率损失
            x = Variable(img.data.clone(), requires_grad=False)

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            pixel_loss_temp = l1_loss(outputs, x)
            ssim_loss_temp = ssim_loss(outputs, x, normalize=True)
            ssim_loss_value += (1 - ssim_loss_temp)
            pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            # total loss
            total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
            total_loss.backward()
            optimizer.step()

            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_pixel_loss / args.log_interval,
                                  all_ssim_loss / args.log_interval,
                                  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
                count_loss = count_loss + 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.

    # pixel loss
    loss_data_pixel = np.array(Loss_pixel)
    loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                         args.ssim_path[i] + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})

    # SSIM loss
    loss_data_ssim = np.array(Loss_ssim)
    loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                         args.ssim_path[i] + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
    # all loss
    loss_data_total = np.array(Loss_all)
    loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
                         args.ssim_path[i] + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'loss_total': loss_data_total})
    # save model
    SwinFuse_model.eval()
    SwinFuse_model.cpu()
    save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(SwinFuse_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    main()

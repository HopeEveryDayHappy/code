# test phase
import os
import torch
import sys
sys.path.append('/kaggle/input/swingfusion1/SwinFuse-main')
print(sys.path)
from torch.autograd import Variable
from net import SwinFuse
import utils
from args_fusion import args
import numpy as np
import time
import cv2


def load_model(path, in_chans, out_chans):

    SwinFuse_model = SwinFuse(in_chans=in_chans, out_chans=out_chans)
    SwinFuse_model.load_state_dict(torch.load(path), False)
    print(SwinFuse_model.parameters())
    para = sum([np.prod(list(p.size())) for p in SwinFuse_model.parameters()])
    print(para)
    type_size = 4
    print('Model {} : params: {:4f}M'.format(SwinFuse_model._get_name(), para * type_size / 1000 / 1000))

    SwinFuse_model.eval()
    SwinFuse_model.cpu()
    # SwinFuse_model.cuda()

    return SwinFuse_model


def run_demo(model, infrared_path, visible_path, output_path_root, index, f_type):
    img_ir, h, w, c = utils.get_test_images(infrared_path)
    img_vi, h, w, c = utils.get_test_images(visible_path)
    # print('test33_img_ir{} h{}  w{}  c{} '.format(img_ir.shape, h, w, c))
    # print('test34_img_vi{} h{}  w{}  c{} '.format(img_vi.shape, h, w, c))
    if c == 0:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        # encoder
        tir3 = model.encoder(img_ir)
        tvi3 = model.encoder(img_vi)
        # fusion
        f = model.fusion(tir3, tvi3, f_type)


        # # encoder1
        # tir1 = model.encoder1(img_ir)
        # tvi1 = model.encoder1(img_vi)
        # # fusion1
        # f1 = model.fusion(tir1, tvi1, f_type)
        # tir1=tir1+f1
        # tvi1=tvi1+f1
        #
        # # encoder2
        # tir2 = model.encoder2(tir1)
        # tvi2 = model.encoder2(tvi1)
        # # fusion2
        # f2 = model.fusion(tir2, tvi2, f_type)
        # tir2 = tir2 + f2
        # tvi2 = tvi2 + f2
        # # encoder3
        # tir3 = model.encoder3(tir2)
        # tvi3 = model.encoder3(tvi2)
        # # fusion2
        # f = model.fusion(tir3, tvi3, f_type)



        # decoder
        img_fusion = model.up_x4(f)
        img_fusion = ((img_fusion / 2) + 0.5) * 255
    else:
        img_fusion_blocks = []
        for i in range(c):
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)
            print(img_ir_temp.shape)
            # encoder
            #
            # print('第%d张图片的结果',i)
            # tir3 = model.encoder(img_ir_temp)
            # tvi3 = model.encoder(img_vi_temp)
            # # fusion
            # f = model.fusion(tir3, tvi3, f_type)
            # print('融合后的结果大小',f.shape)

            # encoder1
            # print('img_ir_temp.shape_test93->',img_ir_temp.shape)
            # print('img_vi_temp.shape_test94->', img_vi_temp.shape)

            tir1 = model.encoder1(img_ir_temp)
            tvi1 = model.encoder1(img_vi_temp)
            tir_u = model.UNet(img_ir_temp)
            tvi_u = model.UNet(img_vi_temp)


            # print('tir1.shape_test95->',tir1.shape)
            # print('tvi1.shape_test96->', tvi1.shape)
            # fusion1
            f1 = model.fusion(tir1, tvi1, f_type)

            tir1 = tir1 +tir_u + f1
            tvi1 = tvi1 +tvi_u+ f1
            tir1=model.transformerback(tir1)
            print('tir1',tir1.shape)
            tvi1=model.transformerback(tvi1)

            # encoder2
            tir2 = model.encoder2(tir1)
            tvi2 = model.encoder2(tvi1)
            # fusion2
            f2 = model.fusion(tir2, tvi2, f_type)
            tir2 = tir2 + tir_u+f2
            tvi2 = tvi2 + tvi_u+f2
            tir2 = model.transformerback(tir2)
            print('tir2',tir2.shape)
            tvi2 = model.transformerback(tvi2)
            # encoder3
            tir3 = model.encoder3(tir2)
            tvi3 = model.encoder3(tvi2)
            tir3 = tir3+tir_u
            tvi3 = tvi3+tvi_u
            # print('tir3.shape_test120->', tir3.shape)
            # print('tvi3.shape_test121->', tvi3.shape)
            # fusion2
            f = model.fusion(tir3, tvi3, f_type)
            # print('f.shape_test124->',f.shape)
            # decoder
            img_fusion = model.up_x4(f)
            img_fusion = ((img_fusion / 2) + 0.5) * 255
            img_fusion_blocks.append(img_fusion)
        if 224 < h < 448 and 224 < w < 448:
            img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)
        if 448 < h < 672 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
        if 448 < h < 672 and 672 < w < 896:
            img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)
        if 224 < h < 448 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
        if 672 < h < 896 and 896 < w < 1120:
            img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
        if 0 < h < 224 and 224 < w < 448:
            img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
        if 0 < h < 224 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
        if h == 224 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)
    ############################ multi outputs ##############################################
    output_count = 0
    for img_fusion in img_fusion_list:
        file_name = 'fusion' + '_' + str(index) + '_swinfuse_' + str(output_count) + '_' + 'f_type' + '.png'
        output_path = output_path_root + file_name
        output_count += 1
        # save images
        utils.save_image_test(img_fusion, output_path)
        print(output_path)


def main():
    # run demo
    test_path = "./data/TNO/"
    # test_ir_path = "D:/Transformer  224Unet/imgs road sence/thermal/"
    # test_vis_path = "D:/Transformer  224Unet/imgs road sence/visual/"
    # test_ir_path = "D:/Transformer  224Unet/INO_TreesAndRunner/INO_TreesAndRunner_T/"
    # test_vis_path = "D:/Transformer  224Unet/INO_TreesAndRunner/INO_TreesAndRunner_Gray/"
    # test_ir_path = "D:/Transformer  224Unet/video/thermal/"
    # test_vis_path = "D:/Transformer  224Unet/video/visual/"

    network_type = 'SwinFuse'
    fusion_type = ['l1_mean']

    output_path = './kaggle/working/result'

    # in_c = 3 for RGB imgs; in_c = 1 for gray imgs
    in_chans = 1

    num_classes = in_chans
    mode = 'L'
    model_path = args.model_path_gray

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[1])
        ssim_weight_str = args.ssim_path[1]
        f_type = fusion_type[0]
        #./SwinFuse model/Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model
        #  1   1
        model = load_model(model_path, in_chans, num_classes)
        # begin = time.time()
        # for a in range(10):
        for i in range(21):
        # for i in range(1000, 1221):
        # for i in range(1000, 1040):
            index = i + 1
            infrared_path = test_path + 'IR/' + str(index) + '.png'
            visible_path = test_path + 'VIS/' + str(index) + '.png'
            # infrared_path = test_ir_path + 'roadscene' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'roadscene' + '_' + str(index) + '.png'
            # infrared_path = test_ir_path + 'video' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'video' + '_' +str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, f_type)
        # end = time.time()
        # print("consumption time of generating:%s " % (end - begin))
    print('Done......')



if __name__ == '__main__':
    main()
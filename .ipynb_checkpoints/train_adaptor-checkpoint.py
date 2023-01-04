import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

from dataloader import sceneflow_loader as sf
import networks.Aggregator as Agg
import networks.U_net as un
import networks.feature_extraction as FE
import loss_functions as lf

from torchinfo import summary
# from torchsummary import summary
from thop import profile
from thop import clever_format

import wandb

from dataloader import KITTIloader as kt
from dataloader import KITTI2012loader as kt2012

parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--seed', type=str, default=42)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--epoch', type=int, default=20)
# parser.add_argument('--data_path', type=str, default='/workspace/mnt/e/datasets/sceneflow/')
parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/sceneflow/')
parser.add_argument('--save_path', type=str, default='trained_models/')
parser.add_argument('--load_path', type=str, default='pretrained_models/checkpoint_baseline_8epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--color_transform', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=True)
# parser.add_argument('--data_path_kitti', type=str, default='/workspace/mnt/e/datasets/kitti2015/training/')
parser.add_argument('--data_path_kitti', type=str, default='/root/autodl-tmp/kitti2015/training/')
parser.add_argument('--kitti', type=str, default='2015')


args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

all_limg, all_rimg, all_ldisp, all_rdisp, test_limg, test_rimg, test_ldisp, test_rdisp = sf.sf_loader(
    args.data_path)

trainLoader = torch.utils.data.DataLoader(
    sf.myDataset(all_limg, all_rimg, all_ldisp, all_rdisp,
                 training=True, color_transform=args.color_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)


# fe_model = FE.VGG_Feature(fixed_param=True).eval()  # 只加载了前15层  MACs: 97.543G , params: 3.471M
# MACs: 8.556G , params: 475.904K
fe_model = FE.CSPDarknet_Feature(fixed_param=True).eval()
model = un.U_Net_v4(img_ch=128, output_ch=64).train()
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
agg_model = Agg.PSMAggregator(args.max_disp, udc=True).eval()

summary(fe_model, (1, 3, 512, 256))
# print(fe_model)
# summary(model, input_size=(1, 256, 160, 160))
# print(model)
# summary(agg_model, input_size=(1, 64, 160, 160))
# print(agg_model)

# input = torch.randn(1, 3, 512, 256).cuda()
# macs, params = profile(fe_model, (input,))
# macs, params = clever_format([macs, params], "%.3f")
# print('MACs:', macs ,', params:', params)

if cuda:
    fe_model = nn.DataParallel(fe_model.cuda())
    model = nn.DataParallel(model.cuda())
    agg_model = nn.DataParallel(agg_model.cuda())

agg_model.load_state_dict(torch.load(args.load_path)[
                          'net'])  # 加载model（PSMAggregator）
for p in agg_model.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

wandb.init(entity="whd", project="Graft-PSMNet")
# Re-run the model without restarting the runtime, unnecessary after our next release
wandb.watch_called = False
# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
# input batch size for training (default: 64)
config.batch_size = args.batch_size
# number of epochs to train (default: 10)
config.epochs = args.epoch
# config.lr = 0.1               # learning rate (default: 0.01)
config.no_cuda = args.no_cuda         # disables CUDA training
config.seed = args.seed                # random seed (default: 42)
# config.log_interval = 1     # how many batches to wait before logging training status

wandb.watch(model, log="all")


def train(imgL, imgR, gt_left, gt_right):
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    gt_left = torch.FloatTensor(gt_left)
    gt_right = torch.FloatTensor(gt_right)

    if cuda:
        imgL, imgR, gt_left, gt_right = imgL.cuda(
        ), imgR.cuda(), gt_left.cuda(), gt_right.cuda()

    optimizer.zero_grad()

    # 对所有包裹的计算操作进行分离。但是torch.no_grad()将会使用更少的内存，因为从包裹的开始，就表明不需要计算梯度了，因此就不需要保存中间结果。
    # requires_grad控制梯度计算，eval控制dropout层停止drop和BN层停止计算均值和方差，但无法控制梯度计算，因此在eval模式下，再加个with_no_grad可以节省计算
    with torch.no_grad():
        left_fea = fe_model(imgL)
        right_fea = fe_model(imgR)

    agg_left_fea = model(left_fea)
    agg_right_fea = model(right_fea)

    loss1, loss2 = agg_model(
        agg_left_fea, agg_right_fea, gt_left, training=True)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)
    loss = 0.1 * loss1 + loss2
    # loss = loss1

    wandb.log({
        "loss1": loss1,
        "loss2": loss2,
        "loss_total": loss})

    loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item()  # 取出单元素张量的元素值并返回该值


def eval_kitti(args, fe_model, model, agg_model):
    model.eval()

    if args.kitti == '2015':
        all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt.kt_loader(
            args.data_path_kitti)
    else:
        all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(
            args.data_path_kitti)

    test_limg = all_limg + test_limg
    test_rimg = all_rimg + test_rimg
    test_ldisp = all_ldisp + test_ldisp

    pred_mae = 0
    pred_op = 0
    for i in trange(len(test_limg)):
        limg = Image.open(test_limg[i]).convert('RGB')
        rimg = Image.open(test_rimg[i]).convert('RGB')

        w, h = limg.size
        m = 16
        wi, hi = (w // m + 1) * m, (h // m + 1) * m
        limg = limg.crop((w - wi, h - hi, w, h))
        rimg = rimg.crop((w - wi, h - hi, w, h))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        limg_tensor = transform(limg)
        rimg_tensor = transform(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        disp_gt = Image.open(test_ldisp[i])
        disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
        gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            left_fea = fe_model(limg_tensor)
            right_fea = fe_model(rimg_tensor)

            left_fea = model(left_fea)
            right_fea = model(right_fea)

            pred_disp = agg_model(left_fea, right_fea,
                                  gt_tensor, training=False)
            pred_disp = pred_disp[:, hi - h:, wi - w:]

        predict_np = pred_disp.squeeze().cpu().numpy()

        op_thresh = 3
        mask = (disp_gt > 0) & (disp_gt < args.max_disp)
        error = np.abs(predict_np * mask.astype(np.float32) -
                       disp_gt * mask.astype(np.float32))

        pred_error = np.abs(
            predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
        pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
        pred_mae += np.mean(pred_error[mask])

    print(pred_mae / len(test_limg))
    print(pred_op / len(test_limg))
    
    wandb.log({
        "pred_mae": pred_mae / len(test_limg),
        "pred_op": pred_op / len(test_limg)})

    model.train()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = 0.00075  # 8  0.001  
    else:
        lr = 0.000075  # 8  0.0001  
    # print(lr)

    wandb.log({
        "lr": lr,
        "epoch": epoch})

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    # start_total_time = time.time()
    start_epoch = 1

    # checkpoint = torch.load('trained_ft_CA_8.12/checkpoint_3_DA.tar')
    # agg_model.load_state_dict(checkpoint['net'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epoch + start_epoch):
        print('This is %d-th epoch' % (epoch))
        total_train_loss1 = 0
        total_train_loss2 = 0
        adjust_learning_rate(optimizer, epoch)
            
        for batch_id, (imgL, imgR, disp_L, disp_R) in enumerate(tqdm(trainLoader)):
            train_loss1, train_loss2 = train(imgL, imgR, disp_L, disp_R)
            total_train_loss1 += train_loss1
            total_train_loss2 += train_loss2
        avg_train_loss1 = total_train_loss1 / len(trainLoader)
        avg_train_loss2 = total_train_loss2 / len(trainLoader)
        print('Epoch %d average training loss1 = %.3f, average training loss2 = %.3f' %
              (epoch, avg_train_loss1, avg_train_loss2))

        if (epoch % 1 == 0) and args.eval:
            eval_kitti(args, fe_model, model, agg_model)
        
        state = {'fa_net': model.state_dict(),
                 'net': agg_model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + \
            'checkpoint_adaptor_{}epoch.tar'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()

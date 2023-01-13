from torch.utils.data.distributed import DistributedSampler
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
import math

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


parser = argparse.ArgumentParser(description='GraftNet')
# parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3')
parser.add_argument('--seed', type=str, default=42)
parser.add_argument('--batch_size', type=int, default=8)  # 所有卡的batchsize之和
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--epoch', type=int, default=10)
# parser.add_argument('--data_path', type=str, default='/workspace/mnt/e/datasets/sceneflow/')
parser.add_argument('--data_path', type=str,
                    default='/root/autodl-tmp/sceneflow/')
parser.add_argument('--save_path', type=str, default='trained_models/')
parser.add_argument('--load_path', type=str,
                    default='pretrained_models/checkpoint_baseline_8epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--color_transform', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=True)
# parser.add_argument('--data_path_kitti', type=str, default='/workspace/mnt/e/datasets/kitti2015/training/')
parser.add_argument('--data_path_kitti', type=str,
                    default='/root/autodl-tmp/kitti2015/training/')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument(
    "--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

torch.manual_seed(args.seed)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(args.seed)

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
num_gpus = torch.cuda.device_count()

all_limg, all_rimg, all_ldisp, all_rdisp, test_limg, test_rimg, test_ldisp, test_rdisp = sf.sf_loader(
    args.data_path)

train_datasets = sf.myDataset(all_limg, all_rimg, all_ldisp, all_rdisp,
                              training=True, color_transform=args.color_transform)
train_sampler = DistributedSampler(train_datasets)
card_batch_size = args.batch_size // num_gpus
trainLoader = torch.utils.data.DataLoader(
    dataset=train_datasets, sampler=train_sampler, batch_size=card_batch_size, shuffle=False, num_workers=4, pin_memory=True)


# fe_model = FE.VGG_Feature(fixed_param=True).eval()  # 只加载了前15层
# fe_model = FE.CSPDarknet_Feature(fixed_param=True).eval()
fe_model = FE.Res50().eval()
model = un.U_Net_v4(img_ch=256, output_ch=64).train()
# print('Number of training model parameters: {}'.format(
#     sum([p.data.nelement() for p in model.parameters()])))
agg_model = Agg.PSMAggregator(args.max_disp, udc=True).eval()
for p in agg_model.parameters():
    p.requires_grad = False

# agg_model.load_state_dict(torch.load(args.load_path)[
#                             'net'])  # 加载model（PSMAggregator）
if args.local_rank == 0:
    model_dict = agg_model.state_dict()
    pretrained_dict = torch.load(args.load_path, map_location=device)['net']
    load_key, no_load_key, temp_dict = [], [], {}
    # print('model_dict')
    # print(model_dict.keys())
    # print('pretrained_dict')
    # print(pretrained_dict.keys())
    for k, v in pretrained_dict.items():
        k = k[7:]
        # print(k)
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    agg_model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[
          :500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[
          :500], "……\nFail To Load Key num:", len(no_load_key))

# checkpoint = torch.load(args.load_path, map_location = device)
# agg_model.load_state_dict({k[7:]: v for k, v in checkpoint['net'].items()},strict=True)


# summary(fe_model, (1, 3, 512, 256))
# print(fe_model)
# summary(model, input_size=(1, 256, 160, 160))
# print(model)
# summary(agg_model, input_size=(1, 64, 160, 160))
# print(agg_model)

# input = torch.randn(1, 3, 512, 256).cuda()
# macs, params = profile(fe_model, (input,))
# macs, params = clever_format([macs, params], "%.3f")
# print('MACs:', macs ,', params:', params)
fe_model.to(device)
# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
model.to(device)
agg_model.to(device)

if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                broadcast_buffers=False)

# if torch.distributed.get_rank() == 0:
#     # agg_model.load_state_dict(torch.load(args.load_path)[
#     #                         'net'])  # 加载model（PSMAggregator）
#     checkpoint = torch.load(args.load_path)
#     model.load_state_dict({k: v for k, v in checkpoint['net'].items()},strict=True)
#     for p in agg_model.parameters():
#         p.requires_grad = False

# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.937, weight_decay=5e-4)

if args.local_rank == 0:
    wandb.init(entity="whd", project="Graft-PSMNet")
    wandb.watch_called = False
    config = wandb.config   
    config.batch_size = args.batch_size
    config.num_gpus = num_gpus
    config.lr = args.lr        
    config.epochs = args.epoch
    config.seed = args.seed 
    wandb.watch(model, log="all")


scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast


def train(imgL, imgR, gt_left, gt_right, epoch):
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

    with autocast():
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

    if args.local_rank == 0:
        wandb.log({
            "loss1": loss1,
            "loss2": loss2,
            "loss_total": loss,
            "epoch": epoch})

    # loss.backward()
    # optimizer.step()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss1.item(), loss2.item()  # 取出单元素张量的元素值并返回该值


def eval_kitti(args, fe_model, model, agg_model, epoch):
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

        with autocast():
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
        "pred_op": pred_op / len(test_limg),
        "epoch": epoch})

    model.train()


def adjust_learning_rate(optimizer, epoch, lr_init, ratio = 0.1):
    if epoch <= 8:
        lr = lr_init  # 8  0.001
    else:
        lr = lr_init * ratio  # 8  0.0001
    # print(lr)

    if args.local_rank == 0:
        wandb.log({
            "lr": lr,
            "epoch": epoch})

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cosine(optimizer, epoch_now, epoch_total, lr_init, lr_ratio=0.1):
    warmup_epoch = math.ceil(epoch_total * 0.1)
    if epoch_now <= warmup_epoch and warmup_epoch == 1:
        lr = lr_init * 0.25 
    elif epoch_now <= warmup_epoch and warmup_epoch > 1:
        lr = lr_init * epoch_now / warmup_epoch
    else:
        lr = lr_init * lr_ratio + (lr_init-lr_init * lr_ratio)*(1 + math.cos(
            math.pi * (epoch_now - warmup_epoch) / (epoch_total - warmup_epoch))) / 2
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if args.local_rank == 0:
        wandb.log({
            "lr": lr,
            "epoch": epoch_now})


def main():

    # start_total_time = time.time()
    start_epoch = 1
    # if torch.distributed.get_rank() == 0:
    #     checkpoint = torch.load('trained_models/checkpoint_adaptor_2epoch.tar')
    #     agg_model.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epoch + 1):
        if args.local_rank == 0:
            print('This is %d-th epoch' % (epoch))
        total_train_loss1 = 0
        total_train_loss2 = 0
        # adjust_learning_rate(optimizer, epoch, args.lr)
        adjust_learning_rate_cosine(optimizer, epoch, args.epoch, args.lr)
        train_sampler.set_epoch(epoch)

        for batch_id, (imgL, imgR, disp_L, disp_R) in enumerate(tqdm(trainLoader)):
            train_loss1, train_loss2 = train(imgL, imgR, disp_L, disp_R, epoch)
            total_train_loss1 += train_loss1
            total_train_loss2 += train_loss2
        avg_train_loss1 = total_train_loss1 / len(trainLoader)
        avg_train_loss2 = total_train_loss2 / len(trainLoader)

        if args.local_rank == 0 and args.eval:
            eval_kitti(args, fe_model, model, agg_model, epoch)

        if args.local_rank == 0:
            print('Epoch %d average training loss1 = %.3f, average training loss2 = %.3f' %
                  (epoch, avg_train_loss1, avg_train_loss2))

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

    wandb.finish()

if __name__ == '__main__':
    main()

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from dataloader import sceneflow_loader as sf
import networks.Aggregator as Agg
import networks.U_net as un
import networks.feature_extraction as FE
import loss_functions as lf

from torchinfo import summary
# from torchsummary import summary
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description='GraftNet')
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--seed', type=str, default=42)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/workspace/mnt/e/datasets/sceneflow/')
parser.add_argument('--save_path', type=str, default='trained_models/')
parser.add_argument('--load_path', type=str, default='trained_models/checkpoint_baseline_8epoch.tar')
parser.add_argument('--max_disp', type=int, default=192)
parser.add_argument('--color_transform', action='store_true', default=False)
args = parser.parse_args()

if not args.no_cuda:
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

all_limg, all_rimg, all_ldisp, all_rdisp, test_limg, test_rimg, test_ldisp, test_rdisp = sf.sf_loader(args.data_path)

trainLoader = torch.utils.data.DataLoader(
    sf.myDataset(all_limg, all_rimg, all_ldisp, all_rdisp, training=True, color_transform=args.color_transform),
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)


# fe_model = FE.VGG_Feature(fixed_param=True).eval()  # 只加载了前15层  MACs: 97.543G , params: 3.471M
fe_model = FE.CSPDarknet_Feature(fixed_param=True).eval() #MACs: 8.556G , params: 475.904K
model = un.U_Net_v4(img_ch=128, output_ch=64).train()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
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

agg_model.load_state_dict(torch.load(args.load_path)['net'])  #加载model（PSMAggregator）
for p in agg_model.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))


def train(imgL, imgR, gt_left, gt_right):
    imgL = torch.FloatTensor(imgL)
    imgR = torch.FloatTensor(imgR)
    gt_left = torch.FloatTensor(gt_left)
    gt_right = torch.FloatTensor(gt_right)

    if cuda:
        imgL, imgR, gt_left, gt_right = imgL.cuda(), imgR.cuda(), gt_left.cuda(), gt_right.cuda()

    optimizer.zero_grad()

    # 对所有包裹的计算操作进行分离。但是torch.no_grad()将会使用更少的内存，因为从包裹的开始，就表明不需要计算梯度了，因此就不需要保存中间结果。
    # requires_grad控制梯度计算，eval控制dropout层停止drop和BN层停止计算均值和方差，但无法控制梯度计算，因此在eval模式下，再加个with_no_grad可以节省计算
    with torch.no_grad():
        left_fea = fe_model(imgL)
        right_fea = fe_model(imgR)

    agg_left_fea = model(left_fea)
    agg_right_fea = model(right_fea)

    loss1, loss2 = agg_model(agg_left_fea, agg_right_fea, gt_left, training=True)
    loss1 = torch.mean(loss1)
    loss2 = torch.mean(loss2)
    loss = 0.1 * loss1 + loss2
    # loss = loss1

    loss.backward()
    optimizer.step()

    return loss1.item(), loss2.item() # 取出单元素张量的元素值并返回该值


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 10:
        lr = 0.001
    else:
        lr = 0.0001
    # print(lr)
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

        state = {'fa_net': model.state_dict(),
                 'net': agg_model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_model_path = args.save_path + 'checkpoint_adaptor_{}epoch.tar'.format(epoch)
        torch.save(state, save_model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


import os
from PIL import Image
from dataloader import readpfm as rp
# import dataloader.preprocess
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random

IMG_EXTENSIONS= [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# filepath = '/home/whd/Graft-PSMNet/SceneFlow/'
def sf_loader(filepath):
    # 确定filepath下文件夹数目，SceneFlow共六个文件夹
    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    # 数据集名称_frames_cleanpass
    image = [img for img in classes if img.find('frames_cleanpass') > -1]
    # 数据集名称_disparity
    disparity = [disp for disp in classes if disp.find('disparity') > -1]

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_right_disp = []


    # monkaa数据集↓  ######################################## 
    monkaa_img = filepath + [x for x in image if 'monkaa' in x][0]
    monkaa_disp = filepath + [x for x in disparity if 'monkaa' in x][0]
    monkaa_dir = os.listdir(monkaa_img)
    for dd in monkaa_dir: # 数据集文件夹内包含很多子文件夹
        left_path = monkaa_img + '/' + dd + '/left/'
        right_path = monkaa_img + '/' + dd + '/right/'
        disp_path = monkaa_disp + '/' + dd + '/left/'
        rdisp_path = monkaa_disp + '/' + dd + '/right/'

        left_imgs = os.listdir(left_path)
        for img in left_imgs:
            img_path = os.path.join(left_path, img)
            if is_image_file(img_path):
                all_left_img.append(img_path)
                all_right_img.append(os.path.join(right_path, img))
                all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')
                all_right_disp.append(rdisp_path + img.split(".")[0] + '.pfm')


    # flyingthings3d数据集↓  ######################################## 
    flying_img = filepath + [x for x in image if 'flying' in x][0]
    flying_disp = filepath + [x for x in disparity if 'flying' in x][0]
    fimg_train = flying_img + '/TRAIN/'  # 数据集内就是这样，分为训练、测试
    fimg_test = flying_img + '/TEST/'
    fdisp_train = flying_disp + '/TRAIN/'
    fdisp_test = flying_disp + '/TEST/'
    fsubdir = ['A', 'B', 'C'] # 训练/测试内分为ABC三部分，每部分内700+文件夹

    for dd in fsubdir: # 训练
        imgs_path = fimg_train + dd + '/'
        disps_path = fdisp_train + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'
            rdisp_path = disps_path + cc + '/right/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    all_left_img.append(img_path)
                    all_right_img.append(os.path.join(right_path, img))
                    all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')
                    all_right_disp.append(rdisp_path + img.split(".")[0] + '.pfm')

    for dd in fsubdir: # 测试
        imgs_path = fimg_test + dd + '/'
        disps_path = fdisp_test + dd + '/'
        imgs = os.listdir(imgs_path)
        for cc in imgs:
            left_path = imgs_path + cc + '/left/'
            right_path = imgs_path + cc + '/right/'
            disp_path = disps_path + cc + '/left/'
            rdisp_path = disps_path + cc + '/right/'

            left_imgs = os.listdir(left_path)
            for img in left_imgs:
                img_path = os.path.join(left_path, img)
                if is_image_file(img_path):
                    test_left_img.append(img_path)
                    test_right_img.append(os.path.join(right_path, img))
                    test_left_disp.append(disp_path + img.split(".")[0] + '.pfm')
                    test_right_disp.append(rdisp_path + img.split(".")[0] + '.pfm')
    

    # driving数据集↓  ######################################## 
    driving_img = filepath + [x for x in image if 'driving' in x][0]
    driving_disp = filepath + [x for x in disparity if 'driving' in x][0]
    dsubdir1 = ['15mm_focallength', '35mm_focallength']  # 第一级
    dsubdir2 = ['scene_backwards', 'scene_forwards'] # 两种焦距都包含前景后景
    dsubdir3 = ['fast', 'slow'] # 前后景都包含两种速度
    for d in dsubdir1:
        img_path1 = driving_img + '/' + d + '/'
        disp_path1 = driving_disp + '/' + d + '/'
        for dd in dsubdir2:
            img_path2 = img_path1 + dd + '/'
            disp_path2 = disp_path1 + dd + '/'
            for ddd in dsubdir3:
                img_path3 = img_path2 + ddd + '/'
                disp_path3 = disp_path2 + ddd + '/'

                left_path = img_path3 + 'left/'
                right_path = img_path3 + 'right/'
                disp_path = disp_path3 + 'left/'
                rdisp_path = disp_path3 + 'right/'

                left_imgs = os.listdir(left_path)
                for img in left_imgs:
                    img_path = os.path.join(left_path, img)
                    if is_image_file(img_path):
                        all_left_img.append(img_path)
                        all_right_img.append(os.path.join(right_path, img))
                        all_left_disp.append(disp_path + img.split(".")[0] + '.pfm')
                        all_right_disp.append(rdisp_path + img.split(".")[0] + '.pfm')

    return all_left_img, all_right_img, all_left_disp, all_right_disp, \
           test_left_img, test_right_img, test_left_disp, test_right_disp


def img_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


def random_transform(left_img, right_img):
    if np.random.rand(1) <= 0.2:
        left_img = transforms.Grayscale(num_output_channels=3)(left_img)
        right_img = transforms.Grayscale(num_output_channels=3)(right_img)
    else:
        left_img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.1)(left_img)
        right_img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.1)(right_img)
    return left_img, right_img

# 继承时需要重写方法 __len__ 和 __getitem__
# __len__ 是提供数据集大小的方法， __getitem__ 是可以通过索引号找到数据的方法
class myDataset(data.Dataset):

    def __init__(self, left, right, left_disp, right_disp, training, imgloader=img_loader, dploader = disparity_loader,
                 color_transform = False):
        self.left = left
        self.right = right
        self.disp_L = left_disp
        self.disp_R = right_disp
        self.imgloader = imgloader
        self.dploader = dploader
        self.training = training
        self.img_transorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.color_transform = color_transform

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = self.disp_R[index]

        left_img = self.imgloader(left)
        right_img = self.imgloader(right)
        dataL, _ = self.dploader(disp_L)
        # np.ascontiguousarray将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR, _ = self.dploader(disp_R)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        if self.training: # 训练时增强
            w, h = left_img.size
            tw, th = 512, 256
            x1 = random.randint(0, w - tw) 
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1+tw, y1+th))  # 随机硬切割为512x256
            right_img = right_img.crop((x1, y1, x1+tw, y1+th)) # 随机硬切割为512x256
            dataL = dataL[y1:y1+th, x1:x1+tw]
            dataR = dataR[y1:y1+th, x1:x1+tw]

            if self.color_transform: # 黑白或者色彩抖动
                left_img, right_img = random_transform(left_img, right_img)

            left_img = self.img_transorm(left_img)  # 标准化
            right_img = self.img_transorm(right_img)# 标准化

            return left_img, right_img, dataL, dataR
        else:# 验证时不增强  分辨率为960x544
            w, h = left_img.size
            left_img = left_img.crop((w-960, h-544, w, h))
            right_img = right_img.crop((w-960, h-544, w, h))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            dataL = Image.fromarray(dataL).crop((w-960, h-544, w, h))
            dataL = np.ascontiguousarray(dataL)
            dataR = Image.fromarray(dataR).crop((w-960, h-544, w, h))
            dataR = np.ascontiguousarray(dataR)

            return left_img, right_img, dataL, dataR

    def __len__(self):
        return len(self.left)






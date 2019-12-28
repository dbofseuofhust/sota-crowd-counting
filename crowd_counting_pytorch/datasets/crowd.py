from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        if not self.method == 'test':
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.*')))

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)

            # resize
            # wd, ht = img.size
            # scale_wd = self.c_size / wd
            # scale_ht = self.c_size / ht
            # img = img.resize((self.c_size, self.c_size))
            # keypoints = np.resize(keypoints, (wd, ht)) / scale_wd / scale_ht

            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)

            # resize
            # wd, ht = img.size
            # scale_wd = self.c_size / wd
            # scale_ht = self.c_size / ht
            # img = img.resize((self.c_size, self.c_size))
            # keypoints = np.resize(keypoints, (wd, ht)) / scale_wd / scale_ht

            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':

            wd, ht = img.size
            max_size = max(wd,ht)
            if max_size > self.c_size:
                scale = self.c_size / max_size
            else:
                scale = 1
            img = img.resize((int(wd*scale), int(ht*scale)),Image.ANTIALIAS)
            img = self.trans(img)
            name = os.path.basename(img_path)
            return img, name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)

        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size

class CrowdSHT(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path

        if not self.method == 'test':
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        else:
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.*')))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = img.resize((self.c_size, self.c_size),Image.ANTIALIAS)
            img = self.trans(img)
            name = os.path.basename(img_path)
            return img, name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)

        if st_size <= self.c_size:
            scale = int(self.c_size / st_size)+1
            wd = int(wd*scale)
            ht = int(ht*scale)
            img = img.resize((wd,ht))
            keypoints = np.resize(keypoints, (wd, ht)) / scale / scale

        # assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size


# 联合数据集 shanghaitech+ucf
class CrowdJoint(data.Dataset):
    def __init__(self, shtecha_root_path, shtechb_root_path, ucf_root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.shtecha_root_path = shtecha_root_path
        self.shtechb_root_path = shtechb_root_path
        self.ucf_root_path = ucf_root_path

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        if not self.method == 'test':
            self.im_list = sorted(glob(os.path.join(self.shtecha_root_path, '*.jpg'))) + sorted(glob(os.path.join(self.shtechb_root_path, '*.jpg'))) + sorted(glob(os.path.join(self.ucf_root_path, '*.jpg')))
        else:
            self.im_list = sorted(glob(os.path.join(self.shtecha_root_path, '*.*'))) + sorted(glob(os.path.join(self.shtechb_root_path, '*.*'))) + sorted(glob(os.path.join(self.ucf_root_path, '*.*')))

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]

        if 'ucf' in img_path or 'UCF' in img_path:
            gd_path = img_path.replace('jpg', 'npy')
        else:
            gd_path = img_path.replace('.jpg', '.npy').replace('images', 'ground_truth')

        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = img.resize((self.c_size, self.c_size), Image.ANTIALIAS)
            img = self.trans(img)
            name = os.path.basename(img_path)
            return img, name

    def train_transform(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)

        # [todo] check here
        if st_size <= self.c_size:
            scale = int(self.c_size / st_size)+1
            wd = int(wd*scale)
            ht = int(ht*scale)
            img = img.resize((wd,ht))
            keypoints = np.resize(keypoints,(wd,ht)) / scale /scale

        # assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = (ratio >= 0.3)

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(target.copy()).float(), st_size




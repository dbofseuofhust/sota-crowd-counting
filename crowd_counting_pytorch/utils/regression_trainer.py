from .trainer import Trainer
from .helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
from ..models.vgg import vgg19
from ..datasets.crowd import Crowd,CrowdJoint
from ..losses.bay_loss import Bay_Loss
from ..losses.post_prob import Post_Prob
from ..models import get_models
from .lr_scheduler import WarmupMultiStepLR
from tqdm import tqdm


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        self.gpus = args.device.strip().split(',')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            # assert self.device_count == 1
            logging.info('using {} gpus'.format(self.args.device))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.use_joint_dataset = args.use_joint_dataset
        self.trainval = args.trainval

        if args.steps is not None:
            self.STEPS = [int(v) for v in args.steps.split(',')]
            logging.info('using mutil step learning rate, [{}].'.format(args.steps))
        else:
            self.STEPS = None

        self.WARMUP_EPOCH = args.warmup_epoch

        self.root_paths = args.data_dir.split(',')

        if self.use_joint_dataset:
            self.datasets = {'train': Crowd([os.path.join(self.root_paths[-1], 'train')]+[os.path.join(v, 'trainval') for v in self.root_paths[:-1]],
                                      args.crop_size,
                                      args.downsample_ratio,
                                      args.is_gray, 'train'),
                             'val': Crowd([os.path.join(self.root_paths[-1], 'val')],
                                      args.crop_size,
                                      args.downsample_ratio,
                                      args.is_gray, 'val'),
                             'trainval': Crowd([os.path.join(v, 'trainval') for v in self.root_paths],
                                      args.crop_size,
                                      args.downsample_ratio,
                                      args.is_gray, 'trainval')}
        else:
            self.datasets = {x: Crowd([os.path.join(v, x) for v in self.root_paths],
                                          args.crop_size,
                                          args.downsample_ratio,
                                          args.is_gray, x) for x in ['train', 'val', 'trainval']}

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                                  collate_fn=(train_collate
                                                  if x in ['train','trainval'] else default_collate),
                                                  batch_size=(args.batch_size
                                                  if x in ['train','trainval'] else 1),
                                                  shuffle=(True if x in ['train','trainval'] else False),
                                                  num_workers=args.num_workers * self.device_count,
                                                  pin_memory=(True if x in ['train','trainval'] else False))
                                    for x in ['train', 'val', 'trainval']}

        # self.model = vgg19()

        self.model = get_models(self.args.model)()

        self.model.to(self.device)

        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # self.scheduler = WarmupMultiStepLR(self.optimizer, self.STEPS, 0.1, 0.1,self.WARMUP_EPOCH, 'linear')

        if self.STEPS is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.STEPS, gamma=0.1, last_epoch=-1)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_mse = checkpoint['best_mse']
                self.best_mae = checkpoint['best_mae']
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

            logging.info('resume from {}, best_mse is {}, best_mae is {}.'.format(args.resume,self.best_mse,self.best_mae))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)

        self.criterion = Bay_Loss(args.use_background, self.device)

        self.save_list = Save_Handle(max_num=args.max_model_num)

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            if self.STEPS is not None:
                self.scheduler.step()
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()

        self.model.train()  # Set model to training mode

        # Iterate over data.
        # for step, (inputs, points, targets, st_sizes) in enumerate(tqdm(self.dataloaders['train'])):

        if self.trainval:
            trainloader = self.dataloaders['trainval']
        else:
            trainloader = self.dataloaders['train']

        for step, (inputs, points, targets, st_sizes) in enumerate(trainloader):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                if 'fpn' in self.args.model:
                    outputs, x2outputs, x3outputs, x4outputs = self.model(inputs)
                    prob_list = self.post_prob(points, st_sizes)
                    loss = sum([self.criterion(prob_list, targets, preds) for preds in [outputs, x2outputs, x3outputs, x4outputs]])
                else:
                    outputs = self.model(inputs)
                    prob_list = self.post_prob(points, st_sizes)
                    loss = self.criterion(prob_list, targets, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

                if step % 20 == 0:
                    print('Train Epoch {} [{}/{}], Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                          .format(self.epoch, step, len(trainloader), epoch_loss.get_avg(),
                                  np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                                  time.time() - epoch_start))

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'best_mse': self.best_mse,
            'best_mae': self.best_mae,
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in tqdm(self.dataloaders['val']):
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))




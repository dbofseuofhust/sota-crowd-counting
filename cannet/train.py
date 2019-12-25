import numpy as np
import time
import torch
import torch.nn as nn
import os
import visdom
import random
import argparse
from tqdm import tqdm as tqdm
from datetime import datetime
import crowd_counting_pytorch as ccp


# https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='directory to save models.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # configuration
    train_image_root = os.path.join(args.data_dir,'part_A_final/train_data', 'images')
    train_dmap_root = os.path.join(args.data_dir,'part_A_final/train_data', 'ground_truth')
    test_image_root = os.path.join(args.data_dir,'part_A_final/test_data', 'images')
    test_dmap_root = os.path.join(args.data_dir,'part_A_final/test_data', 'ground_truth')
    gpu_or_cpu = 'cuda'  # use cuda or cpu
    # lr = 1e-7
    lr = args.lr
    # batch_size = 1
    batch_size = args.batch_size
    momentum = 0.95
    # epochs = 20000
    epochs = args.max_epoch
    # workers = 4
    workers = args.num_workers
    seed = time.time()
    print_freq = 30

    vis = visdom.Visdom()
    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    model = ccp.CANNet().to(device)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=0)
    train_dataset = ccp.CrowdDataset(train_image_root, train_dmap_root, gt_downsample=8, phase='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ccp.CrowdDataset(test_image_root, test_dmap_root, gt_downsample=8, phase='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
    save_dir = os.path.join(args.save_dir, sub_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    min_mae = 10000
    min_epoch = 0
    train_loss_list = []
    epoch_list = []
    test_error_list = []
    for epoch in range(0, epochs):
        # training phase
        model.train()
        epoch_loss = 0
        for i, (img, gt_dmap) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = model(img)
            # calculate loss
            loss = criterion(et_dmap, gt_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(train_loader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(train_loader))
        torch.save(model.state_dict(), os.path.join(save_dir,'model_last.pth'))

        # testing phase
        model.eval()
        mae = 0
        for i, (img, gt_dmap) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = model(img)
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap
        if mae / len(test_loader) < min_mae:
            min_mae = mae / len(test_loader)
            min_epoch = epoch

            # save best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))

        test_error_list.append(mae / len(test_loader))
        print("epoch:" + str(epoch) + " error:" + str(mae / len(test_loader)) + " min_mae:" + str(
            min_mae) + " min_epoch:" + str(min_epoch))
        vis.line(win=1, X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win=2, X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        index = random.randint(0, len(test_loader) - 1)
        img, gt_dmap = test_dataset[index]
        vis.image(win=3, img=img, opts=dict(title='img'))
        vis.image(win=4, img=gt_dmap / (gt_dmap.max()) * 255, opts=dict(title='gt_dmap(' + str(gt_dmap.sum()) + ')'))
        img = img.unsqueeze(0).to(device)
        gt_dmap = gt_dmap.unsqueeze(0)
        et_dmap = model(img)
        et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(win=5, img=et_dmap / (et_dmap.max()) * 255, opts=dict(title='et_dmap(' + str(et_dmap.sum()) + ')'))

    import time

    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
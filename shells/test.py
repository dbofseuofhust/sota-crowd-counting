import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import crowd_counting_pytorch as ccp

args = None

def cal_mae(dataloader,model,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device = torch.device("cuda")

    # model = ccp.vgg19()
    model = ccp.get_models(model)()

    model.load_state_dict(torch.load(model_param_path))
    model.to(device)

    model.eval()
    epoch_minus = []
    with torch.no_grad():
        for i,(inputs, count, name) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)

def estimate_density_map(dataloader,model,model_param_path,index=None,saveroot=None):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device = torch.device("cuda")

    # model = ccp.vgg19()
    model = ccp.get_models(model)()

    model.load_state_dict(torch.load(model_param_path))
    model.to(device)

    if not os.path.exists(saveroot):
        os.makedirs(saveroot)

    model.eval()
    for i,(inputs, count, name) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        if index is not None:
            if i==index:
                # forward propagation
                et_dmap = model(inputs).detach()
                et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
                # plt.imshow(et_dmap,cmap=CM.jet)
                # plt.show()
                plt.imsave(os.path.join(saveroot, '{}.png'.format(name[0])), et_dmap, cmap=CM.jet)
                break
            else:
                continue
        else:
            et_dmap = model(inputs).detach()
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            plt.imsave(os.path.join(saveroot, '{}.png'.format(name[0])),et_dmap,cmap=CM.jet)

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/teddy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/teddy/vgg',
                        help='model directory')
    parser.add_argument('--model', default='/home/teddy/vgg',
                        help='model directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    datasets = ccp.Crowd(os.path.join(args.data_dir, 'test'), 640, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    model_param_path = os.path.join(args.save_dir, 'best_model.pth')
    saveroot = os.path.join(args.save_dir,'visual')

    cal_mae(dataloader,args.model,model_param_path)
    # estimate_density_map(dataloader,args.model,model_param_path,1,saveroot)

    # model = ccp.vgg19()
    # device = torch.device('cuda')
    # model.to(device)
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    # epoch_minus = []
    # for inputs, count, name in dataloader:
    #     inputs = inputs.to(device)
    #     assert inputs.size(0) == 1, 'the batch size should equal to 1'
    #     with torch.set_grad_enabled(False):
    #         outputs = model(inputs)
    #         temp_minu = count[0].item() - torch.sum(outputs).item()
    #         print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
    #         epoch_minus.append(temp_minu)

    # epoch_minus = np.array(epoch_minus)
    # mse = np.sqrt(np.mean(np.square(epoch_minus)))
    # mae = np.mean(np.abs(epoch_minus))
    # log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    # print(log_str)

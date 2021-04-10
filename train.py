#coding=utf-8
from collections import defaultdict
import torch.nn.functional as F
from loss import calc_loss,calc_smoothloss
import time
import torch
from torch.utils.data import Dataset, DataLoader
from utils.lr_scheduler import adjust_learning_rate_poly
from utils.ema import WeightEMA
from utils.label2color import label_img_to_color, diff_label_img_to_color
from data.make_data import GaofenTrain, GaofenVal
from tqdm import tqdm
from evaluate import Evaluator
import numpy as np
import argparse
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import cv2
import os
from tensorboardX import SummaryWriter
from model import UnetIBN
from model.seg_hrnet import hrnet18

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./data')
parser.add_argument('--train_list_path', default='./data/train.txt')
parser.add_argument('--val_list_path', default='./data/val.txt')
# parser.add_argument('--test_list_path', default='./data/val_1w.txt')
parser.add_argument('--backbone', default='resnest50', type=str,help='xception|resnet|resnest101|resnest200|resnest50|resnest26')
parser.add_argument('--n_cls', default=2, type=int)
parser.add_argument('--batchsize', default=4, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--warmup', default=100, type=int)
parser.add_argument('--multiplier', default=100, type=int)
parser.add_argument('--eta_min', default=0.0005, type=float)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--decay_rate', default=0.8, type=float)
parser.add_argument('--decay_epoch', default=200, type=int)
parser.add_argument('--vis_frequency', default=30, type=int)
parser.add_argument('--save_path', default='./results')
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--is_resume', default=False, type=bool)
parser.add_argument('--resume', default='', type=str, help='./results/checkpoint.pth')
args = parser.parse_args()


folder_path = '/backbone={}/warmup={}_lr={}_multiplier={}_eta_min={}_num_epochs={}_batchsize={}'.format(args.backbone,args.warmup,args.lr, args.multiplier, args.eta_min, args.num_epochs,args.batchsize)
isExists = os.path.exists(args.save_path + folder_path)
if not isExists:
    os.makedirs(args.save_path + folder_path)

isExists = os.path.exists(args.save_path +folder_path+'/vis')
if not isExists: os.makedirs(args.save_path + folder_path+'/vis')

isExists = os.path.exists(args.save_path +folder_path+'/log')
if not isExists: os.makedirs(args.save_path + folder_path+'/log')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
global_step = 0
def train_model():
    global global_step
    F_txt = open('./opt_results.txt', 'w')
    evaluator = Evaluator(args.n_cls)
    classes = ['road', 'others']
    writer = SummaryWriter(args.save_path + folder_path + '/log')
    def create_model(ema=False):
        model = hrnet18(pretrained=True).to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()
    ema_model = create_model(ema=True)
    # model = hrnet18(pretrained=True).to(device)
    # model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    ema_optimizer = WeightEMA(model, ema_model, alpha=0.999)
    best_miou = 0.
    best_AA = 0.
    best_OA = 0.
    best_loss = 0.
    lr = args.lr
    epoch_index = 0
    if args.is_resume:
        args.resume = args.save_path + folder_path+'/checkpoint_fwiou.pth'
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            epoch_index = checkpoint['epoch']
            best_miou = checkpoint['miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            F_txt.write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])+'\n')
            # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), file=F_txt)
        else:
            print('EORRO: No such file!!!!!')

    TRAIN_DATA_DIRECTORY = args.root  # '/media/ws/www/IGARSS'
    TRAIN_DATA_LIST_PATH = args.train_list_path  # '/media/ws/www/unet_1/data/train.txt'

    VAL_DATA_DIRECTORY = args.root  # '/media/ws/www/IGARSS'
    VAL_DATA_LIST_PATH = args.val_list_path  # '/media/ws/www/unet_1/data/train.txt'


    dataloaders = {
        "train": DataLoader(GaofenTrain(TRAIN_DATA_DIRECTORY, TRAIN_DATA_LIST_PATH), batch_size=args.batchsize,
                            shuffle=True, num_workers=args.num_workers,pin_memory=True,drop_last=True),
        "val": DataLoader(GaofenVal(VAL_DATA_DIRECTORY, VAL_DATA_LIST_PATH), batch_size=args.batchsize,
                          num_workers=args.num_workers,pin_memory=True)
    }

    evaluator.reset()
    print('config: ' + folder_path)
    print('config: ' + folder_path, file=F_txt,flush=True)
    for epoch in range(epoch_index, args.num_epochs):
        print('Epoch [{}]/[{}] lr={:6f}'.format(epoch + 1, args.num_epochs, lr))
        # F_txt.write('Epoch [{}]/[{}] lr={:6f}'.format(epoch + 1, args.num_epochs, lr)+'\n',flush=True)
        print('Epoch [{}]/[{}] lr={:4f}'.format(epoch + 1, args.num_epochs, lr), file=F_txt,flush=True)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            evaluator.reset()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                ema_model.eval()
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for i, (inputs, labels,edge, _, datafiles) in enumerate(tqdm(dataloaders[phase],ncols=50)):
                inputs = inputs.to(device)
                edge = edge.to(device, dtype = torch.float)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        outputs = model(inputs)
                        outputs[1] = F.interpolate(input=outputs[1], size=(
                            labels.shape[1],labels.shape[2]), mode='bilinear', align_corners=True)
                        loss = calc_loss(outputs, labels, edge, metrics)
                        pred = outputs[1].data.cpu().numpy()
                        pred = np.argmax(pred, axis=1)
                        labels = labels.data.cpu().numpy()
                        evaluator.add_batch(labels, pred)
                    if phase == 'val':
                        outputs = ema_model(inputs)
                        outputs[1] = F.interpolate(input=outputs[1], size=(
                            labels.shape[1],labels.shape[2]), mode='bilinear', align_corners=True)
                        loss = calc_loss(outputs, labels, edge, metrics)
                        pred = outputs[1].data.cpu().numpy()
                        pred = np.argmax(pred, axis=1)
                        labels = labels.data.cpu().numpy()
                        evaluator.add_batch(labels, pred)
                    if phase == 'val' and (epoch+1)%args.vis_frequency==0 and inputs.shape[0]==args.batchsize:
                        for k in range(args.batchsize//2):
                            name = datafiles['name'][k][:-4]

                            writer.add_image('{}/img'.format(name),cv2.cvtColor(cv2.imread(datafiles["img"][k], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),global_step=int((epoch+1)),dataformats='HWC')

                            writer.add_image('{}/gt'.format(name), label_img_to_color(labels[k])[:,:,::-1],global_step=int((epoch+1)),dataformats='HWC')

                            pred_label_img = pred.astype(np.uint8)[k]
                            pred_label_img_color = label_img_to_color(pred_label_img)
                            writer.add_image('{}/mask'.format(name),pred_label_img_color[:,:,::-1],global_step=int((epoch+1)),dataformats='HWC')

                            softmax_pred = F.softmax(outputs[1][k],dim=0)
                            softmax_pred_np = softmax_pred.data.cpu().numpy()
                            probility = softmax_pred_np[0]
                            probility = probility*255
                            probility = probility.astype(np.uint8)
                            probility = cv2.applyColorMap(probility,cv2.COLORMAP_HOT)
                            writer.add_image('{}/prob'.format(name),cv2.cvtColor(probility,cv2.COLOR_BGR2RGB),global_step=int((epoch+1)),dataformats='HWC')
                            # 差分图
                            diff_img = np.ones((pred_label_img.shape[0], pred_label_img.shape[1]), dtype=np.int32)*255
                            mask = (labels[k] != pred_label_img)
                            diff_img[mask] = labels[k][mask]
                            diff_img_color = diff_label_img_to_color(diff_img)
                            writer.add_image('{}/different_image'.format(name), diff_img_color[:, :, ::-1],
                                             global_step=int((epoch + 1)), dataformats='HWC')
                    if phase == 'train':
                        loss.backward()
                        global_step += 1
                        optimizer.step()
                        ema_optimizer.step()
                        adjust_learning_rate_poly(args.lr,optimizer, epoch * len(dataloaders['train']) + i,
                                                       args.num_epochs * len(dataloaders['train']))
                        lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('lr', lr, global_step=epoch * len(dataloaders['train']) + i)
                epoch_samples += 1
            epoch_loss = metrics['loss'] / epoch_samples
            ce_loss = metrics['ce_loss'] / epoch_samples
            ls_loss = metrics['ls_loss'] / epoch_samples
            miou = evaluator.Mean_Intersection_over_Union()
            AA = evaluator.Pixel_Accuracy_Class()
            OA = evaluator.Pixel_Accuracy()
            confusion_matrix = evaluator.confusion_matrix
            if phase == 'val':
                miou_mat = evaluator.Mean_Intersection_over_Union_test()
                writer.add_scalar('val/val_loss', epoch_loss, global_step=epoch)
                writer.add_scalar('val/ce_loss', ce_loss, global_step=epoch)
                writer.add_scalar('val/ls_loss', ls_loss, global_step=epoch)
                #writer.add_scalar('val/val_fwiou', fwiou, global_step=epoch)
                writer.add_scalar('val/val_miou', miou, global_step=epoch)
                for index in range(args.n_cls):
                    writer.add_scalar('class/{}'.format(index+1), miou_mat[index], global_step=epoch)

                print(
                    '[val]------miou: {:4f}, OA:{:4f}, AA: {:4f}, loss: {:4f}'.format( miou, OA, AA,
                                                                                                    epoch_loss))
                print(
                    '[val]------miou: {:4f}, OA:{:4f}, AA: {:4f}, loss: {:4f}'.format(miou, OA, AA,
                                                                                                    epoch_loss),
                    file=F_txt,flush=True)
            if phase == 'train':
                writer.add_scalar('train/train_loss', epoch_loss, global_step=epoch)
                writer.add_scalar('train/ce_loss', ce_loss, global_step=epoch)
                writer.add_scalar('train/ls_loss', ls_loss, global_step=epoch)
                #writer.add_scalar('train/train_fwiou', fwiou, global_step=epoch)
                writer.add_scalar('train/train_miou', miou, global_step=epoch)
                print(
                    '[train]------miou: {:4f}, OA: {:4f}, AA: {:4f}, loss: {:4f}'.format( miou, OA,
                                                                                                       AA, epoch_loss))
                print(
                    '[train]------miou: {:4f}, OA: {:4f}, AA: {:4f}, loss: {:4f}'.format(miou, OA,
                                                                                                       AA, epoch_loss),
                    file=F_txt,flush=True)

            if phase == 'val' and miou > best_miou:
                print("\33[91msaving best model miou\33[0m")
                print("saving best model miou", file=F_txt,flush=True)
                best_miou = miou
                best_OA = OA
                best_AA = AA
                best_loss = epoch_loss
                torch.save({
                    'name': 'resnest50_lovasz_edge_rotate',
                    'epoch': epoch + 1,
                    'state_dict': ema_model.state_dict(),
                    'best_miou': best_miou,
                    'optimizer': optimizer.state_dict(),
                }, args.save_path + folder_path+'/hrnet_ibnUnet+fft+wc0005.pth')
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('{:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60),file=F_txt,flush=True)

    print('[Best val]------miou: {:4f}; OA: {:4f}; AA: {:4f}; loss: {:4f}'.format(best_miou,
                                                                                                best_OA, best_AA,
                                                                                                best_loss))
    print('[Best val]------miou: {:4f}; OA: {:4f}; AA: {:4f}; loss: {:4f}'.format(best_miou,
                                                                                                best_OA, best_AA,
                                                                                                best_loss),file=F_txt,flush=True)
    F_txt.close()
if __name__ == '__main__':
    train_model()

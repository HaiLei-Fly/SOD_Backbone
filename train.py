# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SODLoader
from backbone_vgg import Model
from hybridloss import HybridLoss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')  
    parser.add_argument('--epochs', default=60, help='Number of epochs to train the model for', type=int) 
    parser.add_argument('--bs', default=4, help='Batch size', type=int) 
    parser.add_argument('--lr', default=5e-5, help='Learning Rate', type=float) 
    parser.add_argument('--wd', default=0.0005, help='L2 Weight decay', type=float) 
    parser.add_argument('--img_size', default=352, help='Image size to be used for training', type=int) 
    parser.add_argument('--aug', default=True, help='Whether to use Image augmentation', type=bool) 
    parser.add_argument('--n_worker', default=2, help='Number of workers to use for loading data', type=int) 
    parser.add_argument('--test_interval', default=2, help='Number of epochs after which to test the weights', type=int) 
    parser.add_argument('--save_interval', default=10, help='Number of epochs after which to save the weights. If None, does not save', type=int)
    parser.add_argument('--save_opt', default=False, help='Whether to save optimizer along with model weights or not', type=bool)
    parser.add_argument('--log_interval', default=250, help='Logging interval (in #batches)', type=int) 
    parser.add_argument('--res_mod', default=None, type=str) 
    parser.add_argument('--res_opt', default=None, help='Path to the optimizer to resume from', type=str) 
    parser.add_argument('--use_gpu', default=True, help='Flag to use GPU or not', type=bool) 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser.add_argument('--base_save_path', default='./models', help='Base path for the models to be saved', type=str) 
    parser.add_argument('--weight_bce', default=1.0, help='weight for saliency loss', type=float)
    parser.add_argument('--weight_iou', default=1.0, help='weight for saliency loss', type=float)
    parser.add_argument('--wbce_w0', default=1.0, help='w0 for weighted BCE Loss', type=float)
    parser.add_argument('--wbce_w1', default=1.12, help='w1 for weighted BCE Loss', type=float)

    return parser.parse_args()
 
class Engine:
    def __init__(self, args):
        # epoch
        self.epochs = args.epochs
        # Batch size
        self.bs = args.bs
        # Learning Rate
        self.lr = args.lr
        # L2 Weight decay
        self.wd = args.wd
        self.img_size = args.img_size
        self.aug = args.aug
        self.n_worker = args.n_worker
        self.test_interval = args.test_interval
        self.save_interval = args.save_interval
        self.save_opt = args.save_opt
        self.log_interval = args.log_interval
        self.res_mod_path = args.res_mod
        self.res_opt_path = args.res_opt
        self.use_gpu = args.use_gpu
        
        self.lr_decay_epoch = [8, 16, 24] 
        
        self.weight_bce = args.weight_bce

        self.model_path = args.base_save_path + '/alph-{}'.format(str(self.weight_bce))
        print('Models would be saved at : {}\n'.format(self.model_path))
        
        if not os.path.exists(os.path.join(self.model_path, 'weights')):
            os.makedirs(os.path.join(self.model_path, 'weights'))
        if not os.path.exists(os.path.join(self.model_path, 'optimizers')):
            os.makedirs(os.path.join(self.model_path, 'optimizers'))

        if torch.cuda.is_available():
            self.device = torch.device(device='cuda')
        else:
            self.device = torch.device(device='cpu')

        self.model = Model()
        self.model.to(self.device)
        self.criterion = HybridLoss(device=self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.wd)

        self.print_network(self.model, 'Net Structure')
        
        if self.res_mod_path is not None:
            chkpt = torch.load(self.res_mod_path, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])
            print("Resuming training with checkpoint : {}\n".format(self.res_mod_path))
        if self.res_opt_path is not None:
            chkpt = torch.load(self.res_opt_path, map_location=self.device)
            self.optimizer.load_state_dict(chkpt['optimizer'])
            print("Resuming training with optimizer : {}\n".format(self.res_opt_path))

        self.train_data = SODLoader(mode='train', augment_data=self.aug, target_size=self.img_size)
        self.test_data = SODLoader(mode='test', augment_data=False, target_size=self.img_size)
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.bs, shuffle=True, num_workers=self.n_worker)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.bs, shuffle=False, num_workers=self.n_worker)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():  
            num_params += p.numel()
        total_num = sum(p.numel() for p in model.parameters())    
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))
        print("The number of total parameters: {}".format(total_num))
        print("The number of trainable parameters: {}".format(trainable_num))

    def train(self):
        best_test_mae = float('inf')
        # batch_num = len(self.train_dataloader)
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, (inp_imgs, gt_masks) in enumerate(self.train_dataloader):
                inp_imgs = inp_imgs.to(self.device)
                gt_masks = gt_masks.to(self.device)

                self.optimizer.zero_grad()
                pred_masks, pred_masks1, pred_masks2, pred_masks3, pred_masks4, pred_masks5  = self.model(inp_imgs)
                loss = self.criterion(pred_masks, gt_masks) + self.criterion(pred_masks1, gt_masks) + self.criterion(pred_masks2, gt_masks) + self.criterion(pred_masks3, gt_masks) + self.criterion(pred_masks4, gt_masks) + self.criterion(pred_masks5, gt_masks)
                
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.log_interval == 0:
                    print('TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                          .format(epoch + 1,
                                  batch_idx + 1, len(self.train_dataloader),
                                  (batch_idx + 1) * 100 / len(self.train_dataloader),
                                  loss.item()))
                    
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.wd)

            if epoch % self.test_interval == 0 or epoch % self.save_interval == 0:
                te_avg_loss, te_acc, te_pre, te_rec, te_mae = self.test()
                mod_chkpt = {'epoch': epoch,
                            'test_mae' : float(te_mae),
                            'model' : self.model.state_dict(),
                            'test_loss': float(te_avg_loss),
                            'test_acc': float(te_acc),
                            'test_pre': float(te_pre),
                            'test_rec': float(te_rec)}

                if self.save_opt:
                    opt_chkpt = {'epoch': epoch,
                                'test_mae' : float(te_mae),
                                'optimizer': self.optimizer.state_dict(),
                                'test_loss': float(te_avg_loss),
                                'test_acc': float(te_acc),
                                'test_pre': float(te_pre),
                                'test_rec': float(te_rec)}

                if te_mae <= best_test_mae:
                    best_test_mae = te_mae
                    torch.save(mod_chkpt, self.model_path + '/weights/best-model_epoch-{:03}_mae-{:.4f}_loss-{:.4f}.pth'.
                               format(epoch, best_test_mae, te_avg_loss))
                    if self.save_opt:
                        torch.save(opt_chkpt, self.model_path + '/optimizers/best-opt_epoch-{:03}_mae-{:.4f}_loss-{:.4f}.pth'.
                                   format(epoch, best_test_mae, te_avg_loss))
                    print('Best Model Saved !!!\n')
                    continue
                
                if self.save_interval is not None and epoch % self.save_interval == 0:
                    torch.save(mod_chkpt, self.model_path + '/weights/model_epoch-{:03}_mae-{:.4f}_loss-{:.4f}.pth'.
                               format(epoch, te_mae, te_avg_loss))
                    if self.save_opt:
                        torch.save(opt_chkpt, self.model_path + '/optimizers/opt_epoch-{:03}_mae-{:.4f}_loss-{:.4f}.pth'.
                                   format(epoch, best_test_mae, te_avg_loss))
                    print('Model Saved !!!\n')
                    continue
            print('\n')

    def test(self):
        self.model.eval()
        tot_loss = 0
        # TruePositive + TrueNegative, for accuracy
        tp_fp = 0  
        # TruePositive 
        tp = 0      
        # Number of '1' predictions, for precision
        pred_true = 0  
        # Number of '1's in gt mask, for recall 
        gt_true = 0   
        # List to save mean absolute error of each image  
        mae_list = []   
        fscore, cnt, number   =  0, 0, 256
        mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)

        with torch.no_grad():
            for batch_idx, (inp_imgs, gt_masks) in enumerate(self.test_dataloader, start=1):
                inp_imgs = inp_imgs.to(self.device)
                gt_masks = gt_masks.to(self.device)

                pred_masks, pred_masks1, pred_masks2, pred_masks3, pred_masks4, pred_masks5 = self.model(inp_imgs)
                loss = self.criterion(pred_masks, gt_masks) + self.criterion(pred_masks1, gt_masks) + self.criterion(pred_masks2, gt_masks) + self.criterion(pred_masks3, gt_masks) + self.criterion(pred_masks4, gt_masks) + self.criterion(pred_masks5, gt_masks) 
                
                tot_loss += loss.item()

                tp_fp += (pred_masks.round() == gt_masks).float().sum()
                tp += torch.mul(pred_masks.round(), gt_masks).sum()
                pred_true += pred_masks.round().sum()
                gt_true += gt_masks.sum()

                ae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
                mae_list.extend(ae)
                
                cnt += 1
                
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred_masks >= threshod[i]).float()
                    precision[i] = (temp*gt_masks).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*gt_masks).sum()/(gt_masks.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)

        avg_loss = tot_loss / batch_idx
        accuracy = tp_fp / (len(self.test_data) * self.img_size * self.img_size) 
        precision = tp / pred_true 
        recall = tp / gt_true       
        mae = np.mean(mae_list)
        maxf = fscore.max()/cnt
        print('TEST :: MAE : {:.4f}\tMAXF : {:.4f}\tACC : {:.4f}\tPRE : {:.4f}\tREC : {:.4f}\tAVG-LOSS : {:.4f}\n'.format(mae,
                                                                                             maxf,
                                                                                             accuracy,
                                                                                             precision,
                                                                                             recall,
                                                                                             avg_loss))

        return avg_loss, accuracy, precision, recall, mae 
 
if __name__ == '__main__':
    rt_args = parse_arguments()

    # Driver class
    trainer = Engine(rt_args)
    trainer.train()
    # trainer.test()

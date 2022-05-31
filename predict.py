# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
import time
from skimage import io
from backbone import Model
from dataset import InfDataloader

def parse_arguments(): 
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./media/', help='Path to folder containing images', type=str) 
    parser.add_argument('--imgs_mask_folder', default='./SOD_DATA/alldata/SOD/mask/', help='Path to folder containing images mask', type=str) 
    parser.add_argument('--prediction_folder', default='./Pre/38_Cloud_Train/38-Cloud-Test-20/', help='Path to folder results images', type=str) 
    parser.add_argument('--model_path', default='./model/38_Cloud_Train/best-model_epoch-034_mae-0.0205_loss-0.1703.pth', help='Path to model', type=str) 
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=352, help='Image size to be used', type=int) 
    parser.add_argument('--bs', default=1, help='Batch Size for testing', type=int) 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    return parser.parse_args()

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/((ma-mi) + 1e-8)

	return dn

def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')
 
def predict(args):
    
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')
        
    img_name_list = glob.glob(args.imgs_folder + '*.png')
    
    model = Model()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inf_data = PreDataloader(img_folder=img_name_list, target_size=args.img_size)
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=False, num_workers=1)
    for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _, _, _, _, _  = model(img_tor)
            # pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))
            # normalization
            pred = pred_masks[:,0,:,:]
            pred = normPRED(pred)
            save_output(img_name_list[batch_idx-1], pred, args.prediction_folder)
            del pred_masks

if __name__ == '__main__':
    rt_args = parse_arguments()
    predict(rt_args)


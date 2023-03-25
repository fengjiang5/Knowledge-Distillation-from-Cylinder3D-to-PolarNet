#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import yaml

from cylinder_network.model_builder import cylinder_build
from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset
from network.lovasz_losses import lovasz_softmax
from kd.cross_attn_spconv121 import KD_Part
from kd.weight_mse import Weight_MSE
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

GRANDFA = os.path.dirname(os.path.realpath(__file__))
sys.path.append(GRANDFA)

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+2)
    hist=hist[unique_label+1,:]
    hist=hist[:,unique_label+1]
    return hist

def get_weight(voxel_label, tensor_w):
    B, X, Y, Z = voxel_label.size()
    voxel_weight = tensor_w[voxel_label.to(torch.int64).reshape(-1)]
    voxel_weight = voxel_weight.reshape(B,X,Y,Z)
    return voxel_weight

def main(args):
    data_path = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    compression_model = args.grid_size[2]
    grid_size = args.grid_size
    pytorch_device = torch.device('cuda:0')
    model = args.model
    if model == 'polar':
        fea_dim = 9
        circular_padding = True
    elif model == 'traditional':
        fea_dim = 7
        circular_padding = False

    #prepare miou fun
    unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    content = np.zeros((len(unique_label_str)+1), dtype=np.float32)
    with open("semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map']
    content_all = semkittiyaml['content']
    for key in content_all.keys():
        content[learning_map[key]] += content_all[key]
    loss_w = 1 / (content + 0.001)
    tensor_w = torch.tensor(loss_w).to(pytorch_device)
    #prepare polarnet model
    # 20 classes
    my_BEV_model=BEV_Unet(n_class=len(unique_label)+1, n_height = compression_model, 
                            input_batch_norm = True, dropout = 0.5, dropblock=True,
                            circular_padding = circular_padding)
    my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, 
                        fea_dim = fea_dim, max_pt_per_encode = 256, out_pt_fea_dim = 512, 
                        kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
    if os.path.exists(model_load_path):
        print('Load Polarnet ing...')
        checkpoint = torch.load(model_load_path)
        state = my_model.state_dict()
        match,dismatch = 0,0
        part_load = {}
        for key in checkpoint.keys():
            value = checkpoint[key]
            if key in state and state[key].shape == value.shape:
                part_load[key] = value
                match += 1
            else:
                dismatch += 1
        state.update(part_load)
        my_model.load_state_dict(state)
        # except:
        #     my_model.load_state_dict(torch.load(model_load_path)['model'])
    print('Match:{} Dismatch:{}'.format(match, dismatch))
    my_model.to(pytorch_device)
    
    

    # prepare cylinder3d model
    cylinder_model = cylinder_build()
    if os.path.exists('pretrained/cylinder3d_model_save_backup.pt'):
        print('Load Cylinder3D ing...')
        cylinder_model.load_state_dict(torch.load('pretrained/cylinder3d_model_save_backup.pt'))
        for _, value in cylinder_model.named_parameters():
            value.requires_grad = False
    cylinder_model.to(pytorch_device)
    cylinder_model.eval()

    model_kd = KD_Part(dim_cylinder_list=[64,256,512,128], dim_polar_list=[128,512,256,64])
    # model_kd = KD_Part(dim_cylinder_list=[64], dim_polar_list=[64])
    model_kd = model_kd.to(pytorch_device)
    
    optimizer = optim.Adam([
                {'params':my_model.parameters()},
                {'params':model_kd.parameters()}], lr=0.0005)
    
    loss_fun = torch.nn.CrossEntropyLoss(ignore_index=0, weight=tensor_w)
    loss_func_kd = torch.nn.MSELoss(reduce='mean')
    loss_weight_kd = Weight_MSE()
    #prepare dataset
    train_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'train', return_ref = True,
                                instance_pkl_path='/data1/SemanticKITTI/dataset', finetune=True)
    val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True)
    if model == 'polar':
        train_dataset=spherical_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True,
                                        ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=spherical_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, 
                                      fixed_volume_space = True)
    elif model == 'traditional':
        train_dataset=voxel_dataset(train_pt_dataset, grid_size = grid_size, flip_aug = True, ignore_label = 0,rotate_aug = True, fixed_volume_space = True)
        val_dataset=voxel_dataset(val_pt_dataset, grid_size = grid_size, ignore_label = 0, fixed_volume_space = True)
    train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = train_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = True,
                                                    num_workers = 4)
    val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = val_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

    # training
    epoch=0
    best_val_miou=0
    global_iter = 0
    while epoch < 40:
        loss_list=[]
        pbar_train = tqdm(total=len(train_dataset_loader))
        my_model.train()
        for i_iter,(_,train_vox_label,train_grid,train_pt_labs,train_pt_fea,label_path) in enumerate(train_dataset_loader):
            # training
            batch_size = len(train_grid)
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor=train_vox_label.type(torch.LongTensor).to(pytorch_device)
    
            # forward + backward + optimize
            outputs, polar_midfea = my_model(train_pt_fea_ten,train_grid_ten)
            cylinder_output, cylinder_midfea = cylinder_model(train_pt_fea_ten,train_vox_ten, batch_size=batch_size)
            kd_cylinder, kd_polar = model_kd(cylinder_midfea, polar_midfea)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,ignore=0) +\
                loss_fun(outputs,point_label_tensor)
            
            voxel_weight = get_weight(train_vox_label, tensor_w) / 50
            loss_kd_vpd = loss_func_kd(kd_cylinder[0], kd_polar[0]) +loss_func_kd(kd_cylinder[1], kd_polar[1])+\
            loss_func_kd(kd_cylinder[2], kd_polar[2])+loss_func_kd(kd_cylinder[3], kd_polar[3])
            loss_kd_lwd = loss_weight_kd(kd_cylinder[0], kd_polar[0],voxel_weight)
            # kl loss
            polar_pre = F.softmax(outputs*.1, dim=1)
            cylinder_pre = F.softmax(cylinder_output*.1, dim=1)
            loss_kl = torch.mean(cylinder_pre*torch.log((cylinder_pre+1e-6)/(polar_pre+1e-6)), dim=1)
            loss_kl = loss_kl[loss_kl > 0.01]
            loss_kl = loss_kl[~torch.isnan(loss_kl)]
            loss_kl = torch.mean(loss_kl)
            # loss: segmentation loss
            # loss_kl: logit distillation loss
            # loss_kd_vpd: voxel-to-pillar distillation
            # loss_kd_lwd: label-weight distillation
            loss += loss_kl*1 + loss_kd_vpd * 2 + loss_kd_lwd * 2
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            # zero the parameter gradients
            optimizer.zero_grad()
            pbar_train.update(1)
            global_iter += 1
            
            if global_iter % 5000 == 0:
                # begin validation
                print('start valid')
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    pbar_val = tqdm(total=len(val_dataset_loader), ncols=100)
                    for i_iter_val,(_,val_vox_label,val_grid,val_pt_labs,val_pt_fea,label_path) in enumerate(val_dataset_loader):
                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
                        val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels,_ = my_model(val_pt_fea_ten, val_grid_ten)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,ignore=0) +\
                                loss_fun(predict_labels.detach(),val_label_tensor)
                        predict_labels = torch.argmax(predict_labels,dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count,i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[count,val_grid[count][:,0],
                                                                            val_grid[count][:,1],
                                                                            val_grid[count][:,2]],
                                                                        val_pt_labs[count], unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                        pbar_val.update(1)
                        # if i_iter_val % 5 == 0:
                        #     break
                        # break
                    pbar_val.close()
                print('epoch:{}\n'.format(epoch))
                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str,iou):
                    print('%s : %.2f%%' % (class_name, class_iou*100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label,val_grid,val_pt_fea,val_grid_ten
                
                # save model if performance is improved
                if best_val_miou<val_miou:
                    best_val_miou=val_miou
                    if not os.path.exists(model_save_path):
                        os.mkdir(model_save_path)
                    torch.save(my_model.state_dict(), model_save_path+'/checkpoint.pt')

                print('Current val miou is %.3f while the best val miou is %.3f' %
                    (val_miou,best_val_miou))
                print('Current val loss is %.3f' %
                    (np.mean(val_loss_list)))
                my_model.train()
                # end validation
            # break
        pbar_train.close()
        # end training
        epoch += 1

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data_dir', default='your_dataset_path')
    parser.add_argument('-sp', '--model_save_path', default='your_save_path')
    parser.add_argument('-lp', '--model_load_path', default='your_load_path')
    parser.add_argument('-m', '--model', choices=['polar','traditional'], default='polar', help='training model: polar or traditional (default: polar)')
    parser.add_argument('-s', '--grid_size', nargs='+', type=int, default = [480,360,32], help='grid size of BEV representation (default: [480,360,32])')
    parser.add_argument('--train_batch_size', type=int, default=2, help='batch size for training (default: 1)')
    parser.add_argument('--val_batch_size', type=int, default=2, help='batch size for validation (default: 2)')
    parser.add_argument('--check_iter', type=int, default=4000, help='validation interval (default: 4000)')
    
    args = parser.parse_args()
    if not len(args.grid_size) == 3:
        raise Exception('Invalid grid size! Grid size should have 3 dimensions.')

    print(' '.join(sys.argv))
    print(args)
    main(args)

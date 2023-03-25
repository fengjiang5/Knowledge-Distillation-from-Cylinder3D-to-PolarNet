#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
from pickle import TRUE
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data

instance_classes = [1, 2, 3, 4, 5, 6, 7]
Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3


class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, finetune=False, instance_pkl_path ='data'):
        self.return_ref = return_ref
        self.finetune = finetune
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        thing_class = semkittiyaml['thing_class']
        self.thing_list = [cl for cl, ignored in thing_class.items() if ignored]
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if self.finetune:
                split.append('08')
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')
        
        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()
        self.instance_pkl_path = instance_pkl_path
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)
    
    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            label_path = self.im_idx[index].replace('velodyne','labels')[:-3]+'label'
            sem_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            inst_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=np.uint32),axis=1)
        else:
            label_path = self.im_idx[index].replace('velodyne','labels')[:-3]+'label'
            annotated_data = np.fromfile(label_path, dtype=np.int32).reshape((-1,1))
            sem_data = annotated_data & 0xFFFF #delete high 16 digits binary
            sem_data = np.vectorize(self.learning_map.__getitem__)(sem_data)
            inst_data = annotated_data
        data_tuple = (raw_data[:,:3], sem_data.astype(np.uint8),inst_data)
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        data_tuple += (label_path,)
        return data_tuple

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

class voxel_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
            fixed_volume_space= False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        max_bound = np.percentile(xyz,100,axis = 0)
        min_bound = np.percentile(xyz,0,axis = 0)
        
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        
        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)
        
        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

class spherical_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
               fixed_volume_space= False, max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               scale_aug=False,
                transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                min_rad=-np.pi / 4, max_rad=np.pi / 4, use_tta=False):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        
        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)
        self.use_tta = use_tta
        
        self.polarmix = polarmix


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)
    
  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if self.use_tta:
            data_total = []
            voting = 4
            for idx in range(voting):
                data_single_ori = self.get_single_sample(data, index, idx)
                data_total.append(data_single_ori)
            data_total = tuple(data_total)
            return data_total
        else:
            data_single = self.get_single_sample(data, index)            
            return data_single

  def get_single_sample(self, data, index, vote_idx=0):
        'Generates one sample of data'
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 5:
            xyz,labels,insts,sig,label_path = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        # elif len(data) == 5:
        #     xyz, labels, sig,label_path, origin_len = data
        #     if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')
        
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate
        
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)
        
        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1 
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = polar2cat(voxel_position)
        
        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        # data_tuple = (voxel_position,processed_label)

        # prepare visiblity feature
        # find max distance index in each angle,height pair
        valid_label = np.zeros_like(processed_label,dtype=bool)
        valid_label[grid_ind[:,0],grid_ind[:,1],grid_ind[:,2]] = True
        valid_label = valid_label[::-1]
        max_distance_index = np.argmax(valid_label,axis=0)
        max_distance = max_bound[0]-intervals[0]*(max_distance_index)
        distance_feature = np.expand_dims(max_distance, axis=2)-np.transpose(voxel_position[0],(1,2,0))
        distance_feature = np.transpose(distance_feature,(1,2,0))
        # convert to boolean feature
        distance_feature = (distance_feature>0)*-1.
        distance_feature[grid_ind[:,2],grid_ind[:,0],grid_ind[:,1]]=1.

        data_tuple = (distance_feature,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 4 or len(data) == 5:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)
        
        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea,index)
        return data_tuple
    
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    label_path = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,label_path

def collate_fn_BEV_test(data):    
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    label_path = [d[6] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index,label_path

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

def collate_fn_BEV_tta(data):    

    voxel_label = []

    for da1 in data:
        for da2 in da1:
            voxel_label.append(da2[1])

    #voxel_label.astype(np.int)

    grid_ind_stack = []
    for da1 in data:
        for da2 in da1:
            grid_ind_stack.append(da2[2])



    point_label = []

    for da1 in data:
        for da2 in da1:
            point_label.append(da2[3])

    xyz = []

    for da1 in data:
        for da2 in da1:
            xyz.append(da2[4])
    index = []
    for da1 in data:
        for da2 in da1:
            index.append(da2[5])

    return xyz, xyz, grid_ind_stack, point_label, xyz, index



instance_classes_kitti = [0, 1, 2, 3, 4, 5, 6, 7]

def swap(pt1, pt2, start_angle, end_angle, label1, label2, insts1, insts2):
    # calculate horizontal angle for each point
    yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
    yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

    # select points in sector
    idx1 = np.where((yaw1>start_angle) & (yaw1<end_angle))
    idx2 = np.where((yaw2>start_angle) & (yaw2<end_angle))

    # swap
    pt1_out = np.delete(pt1, idx1, axis=0)
    pt1_out = np.concatenate((pt1_out, pt2[idx2]))
    pt2_out = np.delete(pt2, idx2, axis=0)
    pt2_out = np.concatenate((pt2_out, pt1[idx1]))

    label1_out = np.delete(label1, idx1)
    label1_out = np.concatenate((label1_out, label2[idx2]))
    label2_out = np.delete(label2, idx2)
    label2_out = np.concatenate((label2_out, label1[idx1]))
    
    insts1_out = np.delete(insts1, idx1)
    insts1_out = np.concatenate((insts1_out, insts2[idx2]))
    insts2_out = np.delete(insts2, idx2)
    insts2_out = np.concatenate((insts2_out, insts1[idx1]))
    
    assert pt1_out.shape[0] == label1_out.shape[0]
    assert pt2_out.shape[0] == label2_out.shape[0]
    assert label1_out.shape[0] == insts1_out.shape[0]
    assert label2_out.shape[0] == insts2_out.shape[0]

    return pt1_out, pt2_out, label1_out, label2_out, insts1_out, insts2_out

def rotate_copy(pts, labels, insts,instance_classes, Omega):
    # extract instance points
    pts_inst, labels_inst, insts_inst = [], [], []
    for s_class in instance_classes:
        pt_idx = np.where((labels == s_class))
        pts_inst.append(pts[pt_idx])
        labels_inst.append(labels[pt_idx])
        insts_inst.append(insts[pt_idx])
    pts_inst = np.concatenate(pts_inst, axis=0)
    labels_inst = np.concatenate(labels_inst, axis=0)
    insts_inst = np.concatenate(insts_inst, axis=0)

    # rotate-copy
    pts_copy = [pts_inst]
    labels_copy = [labels_inst]
    insts_copy = [insts_inst]
    for omega_j in Omega:
        rot_mat = np.array([[np.cos(omega_j),
                             np.sin(omega_j), 0],
                            [-np.sin(omega_j),
                             np.cos(omega_j), 0], [0, 0, 1]])
        new_pt = np.zeros_like(pts_inst)
        new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
        new_pt[:, 3] = pts_inst[:, 3]
        pts_copy.append(new_pt)
        labels_copy.append(labels_inst)
        insts_copy.append(insts_inst)
    pts_copy = np.concatenate(pts_copy, axis=0)
    labels_copy = np.concatenate(labels_copy, axis=0)
    insts_copy = np.concatenate(insts_copy, axis=0)
    return pts_copy, labels_copy,insts_copy

def polarmix(pts1, labels1, insts1, pts2, labels2, insts2, alpha, beta, instance_classes, Omega):
    pts_out, labels_out, insts_out = pts1, labels1, insts1
    # swapping
    if np.random.random() < 0.5:
        pts_out, _, labels_out, _, insts_out, _= swap(pts1, pts2, start_angle=alpha, end_angle=beta,
                                                    label1=labels1, label2=labels2,
                                                    insts1=insts1, insts2=insts2)

    # rotate-pasting
    if np.random.random() < 1.0:
        # rotate-copy
        pts_copy, labels_copy, insts_copy = rotate_copy(pts2, labels2, insts2, instance_classes, Omega)
        # paste
        pts_out = np.concatenate((pts_out, pts_copy), axis=0)
        labels_out = np.concatenate((labels_out, labels_copy), axis=0)
        insts_out = np.concatenate((insts_out, insts_copy), axis=0)

    return pts_out, labels_out, insts_out
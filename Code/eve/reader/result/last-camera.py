import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import readcam



def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    anno.cam = line[4]
    anno.norm = line[6]
    return anno

def gazeto3d(gaze):
    assert gaze.size == 2, "The size of gaze must be 2"
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
    gaze_gt[1] = -np.sin(gaze[0])
    gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
    return gaze_gt


def Decode_Dict():
    mapping = edict()
    mapping.ethtrain = Decode_ETH
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

class trainloader(Dataset): 
    def __init__(self, dataset):

        # Read source data
        self.data = edict() 
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        self.data.cam_num = dataset.cams.num
        self.data.cam_params = readcam.cam_params
        

        if isinstance(dataset.label, list):

            for i in dataset.label:
                with open(i) as f: line = f.readlines()
                if dataset.header: line.pop(0)

                self.data.line.extend(line)

        else:

            with open(dataset.label) as f: self.data.line = f.readlines()

            if dataset.header: self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):

        return len(self.data.line)


    def __getitem__(self, idx):

        num = (idx//18)*18

        length = 8
        
        image_nums = np.random.choice(np.arange(9, 18), size=length-1, replace=False)
        image_nums = image_nums + num
        image_nums = np.append(image_nums, idx)
     
        images = []
        labels = []
        cams = []
        poses = []
        names = []

        for i in image_nums:

            # Read souce information
            line = self.data.line[i]
            line = line.strip().split(" ")
            anno = self.data.decode(line)

            # Image
            img = cv2.imread(os.path.join(self.data.root, anno.face))
            img = self.transforms(img)
            img = img.unsqueeze(0)
            images.append(img)

            # Label
            label = np.array(anno.gaze2d.split(",")).astype("float")
            label = gazeto3d(label)
            label = torch.from_numpy(label).type(torch.FloatTensor)
            label = label.unsqueeze(0)
            labels.append(label)

            # Camera rotation. Label = R * prediction
            norm_mat = np.array(anno.norm.split(",")).astype('float')
            norm_mat = np.resize(norm_mat, (3, 3))

            cam_mat = self.data.cam_params[int(anno.cam)-1].rotation

            new_mat = np.dot(norm_mat, cam_mat)
            new_mat = torch.from_numpy(new_mat).type(torch.FloatTensor)
            new_mat = new_mat.unsqueeze(0)
            cams.append(new_mat)

            # Pos.
            z_axis = np.linalg.inv(new_mat)[:, 2].flatten()
            translation = self.data.cam_params[int(anno.cam)-1].translation
            pos = np.concatenate([z_axis, translation], 0)
            pos = torch.from_numpy(pos).type(torch.FloatTensor)
            pos = pos.unsqueeze(0) 
            poses.append(pos)

            # Name
            names.append(anno.name)

        data = edict()
        data.face = torch.cat(images, 0)
        data.cams = torch.cat(cams, 0)
        data.pos = torch.cat(poses, 0)
        data.name = names

        return data, torch.cat(labels, 0)

def loader(source, batch_size, shuffle=True,  num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


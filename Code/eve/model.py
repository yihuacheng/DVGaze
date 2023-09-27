import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from resnet import resnet18
from transformer import Transformer



class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        # basic model

        len_feature = 128

        self.base_model = resnet18(pretrained = True,
                out_feature = len_feature)

        self.transformer = Transformer(feature_num = len_feature, 
                                pos_length = 6)

        self.feed = nn.Linear(len_feature, 2)

    
        self.loss_op_gaze = nn.L1Loss()
        self.loss_op_cons = nn.MSELoss()

    def forward(self, x_in):

        # x_in['face']: [Batch camera channel width length]

        # Get face feature [length, batch, dim]

        faces = x_in['face']


        feature = self.base_model(faces)

        feature = self.transformer(feature, x_in['pos'])

        gaze = self.feed(feature)

        gaze1 = gaze[:, 0, :]

        gaze2 = gaze[:, 1, :]

        return gaze1, gaze2

    def gazeto3d(self, gaze):
        x = -torch.cos(gaze[:, 0]) * torch.sin(gaze[:, 1])
        y = -torch.sin(gaze[:, 0])
        z = -torch.cos(gaze[:, 0]) * torch.cos(gaze[:, 1])
        gaze = torch.stack([x, y, z], dim=1)
        return gaze
        

        
    def loss(self, x_in, label):

        # x_in: [batch, length, dim]
        # Gaze: [batch, 3]

        gaze1, gaze2 = self.forward(x_in)

        loss_op = self.loss_op_gaze

        loss = loss_op(gaze1, label['gaze'][:, 0, :])+\
               loss_op(gaze2, label['gaze'][:, 1, :])


        cams = x_in['cams']

        cam1 = cams[:, 0:3, :]
        cam2 = cams[:, 3:6, :]
        gaze3d1 = self.gazeto3d(gaze1)
        gaze3d2 = self.gazeto3d(gaze2)

        origin3d1 = torch.einsum('ijk,ik->ij', [cam1, gaze3d1])
        origin3d2 = torch.einsum('ijk,ik->ij', [cam2, gaze3d2])
        
        loss2 = self.loss_op_cons(origin3d1, origin3d2)

        return loss + 0.5*loss2

if __name__ == '__main__':


    model = Model().cuda()

    params = list(model.parameters())
    k = 0

    for i in params:
        l = 1
        for j in i.size():
            l*=j
        k+= l

    print(k)

    

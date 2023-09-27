from xml.dom.minidom import parse
import xml.dom.minidom 
import sys
from easydict import EasyDict as edict
import os
import numpy as np


def camparse(filename):
    # Rotation: from 0th Camera to Nth Camera
    # P_N = R*P_0 + T.
    # P_0, coordinates w.r.t 0th CCS.
    # P_N, coordinated w.r.t Nth CCS.
    DOMTree = xml.dom.minidom.parse(filename)

    collection = DOMTree.documentElement

    intrinsic = collection.getElementsByTagName("Camera_Matrix")[0]
    rotation = collection.getElementsByTagName("cam_rotation")[0]
    translation = collection.getElementsByTagName("cam_translation")[0]

    result = edict()
    ins = intrinsic.getElementsByTagName('data')[0].childNodes[0].data
    ins = ins.strip().replace('\n', '')
    ins = ins.split(" ")
    ins = [i  for i in ins if i is not '' and i is not ' ']
    result.intrinsic = np.resize(np.array(ins).astype('float'), (3,3))


    rot = rotation.getElementsByTagName('data')[0].childNodes[0].data
    rot = rot.strip().replace('\n', '')
    rot = rot.split(" ")
    rot = [i  for i in rot if i is not '' and i is not ' ']
    result.rotation = np.resize(np.array(rot).astype('float'), (3,3))

    trans = translation.getElementsByTagName('data')[0].childNodes[0].data
    trans = trans.strip().replace('\n', '')
    trans = trans.split(" ")
    trans = [i  for i in trans if i is not '' and i is not ' ']
    result.translation = np.resize(np.array(trans).astype('float'), (3))

   
    return result

def readall():
    cam_path = '/home/chengyihua/dataset/FaceBased/ETH-Gaze/calibration/cam_calibration'
    filenames = os.listdir(cam_path)
    filenames.sort()

    cams = []
    for filename in filenames:
        cam = camparse(os.path.join(cam_path, filename))
        cams.append(cam)

    return cams

cam_params = readall()
        

if __name__ == '__main__':
    files = readall()
    for i in range(18):
        print(files[i])
    


import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

def gazeto3d(gaze):
    assert gaze.size == 2, "The size of gaze must be 2"
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
    gaze_gt[1] = -np.sin(gaze[0])
    gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
    return gaze_gt
 
def main(train, test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.Model()

        statedict = torch.load(
                        os.path.join(modelpath, 
                            f"Iter_{saveiter}_{train.save.model_name}.pt"), 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )

        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset); accs0 = 0; accs2 = 0; count = 0

        logname = f"{saveiter}.log"

        outfile0 = open(os.path.join(logpath, f"cam0_" + logname), 'w')
        outfile0.write("name results gts acc\n")
 
        outfile2 = open(os.path.join(logpath, f"cam2_" + logname), 'w')
        outfile2.write("name results gts acc\n")
        
       

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                for key in label:
                    if key != 'name': label[key] = label[key].cuda()

                names = data["name"]

                gts0 = label['gaze'][:,0,:]
                gts2 = label['gaze'][:,1,:]
           
                gaze0, gaze2  = net(data)

                gts = gts0
                gazes = gaze0

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1                

                    acc = gtools.angular(gazeto3d(gaze), gazeto3d(gt))
                    accs0 += acc
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)] + [str(acc)]
                    outfile0.write(" ".join(log) + "\n")


                gts = gts2
                gazes = gaze2

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]


                    acc = gtools.angular(gazeto3d(gaze), gazeto3d(gt))
                    accs2 += acc
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)] + [str(acc)]
                    outfile2.write(" ".join(log) + "\n")


            loger = f"[{saveiter}] Total Num: {count}, cam0: {accs0/count} cam2: {accs2/count}"
            outfile0.write(loger)
            outfile2.write(loger)
            print(loger)
        outfile0.close()
        outfile2.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test)

 

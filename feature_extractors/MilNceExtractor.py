import cv2
import math
import time
import numpy as np
import os
import glob
import clip
import torch
from torch.utils.data import DataLoader
import torchvision
import cv2
from tqdm import tqdm 
from PIL import Image
from s3dg import S3D

class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, frames,dim):
        self.frames = frames
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop(dim),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.transform(self.frames[idx])
        
opj = lambda x, y: os.path.join(x, y)

device = "cuda:3" if torch.cuda.is_available() else "cpu"
net = S3D('s3d_dict.npy', 512)
net.load_state_dict(torch.load('s3d_howto100m.pth'))
net.to(device)
net = net.eval()

def extract_and_save(full_path):
    batch_size = 32
    path, dest, vid = '/'.join(full_path.split('/')[:-1]), '/'.join(full_path.split('/')[4:-1]), full_path.split('/')[-1]
    video = cv2.VideoCapture(opj(path, vid))
    frames = []
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        video.release()
    min_dim = min(frames[0].shape[0:2])
    dataset = VideoLoader(frames,min_dim)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_batch = len(data_loader)
    emb = torch.tensor([]).to(device)
    concat_emb = torch.tensor([])
    print("Reading from data loader")
    count = 0
    for i,data in enumerate(data_loader):
        count += 1
        data = data/255.0
        data = data.permute(1,0,2,3)
        if i == num_batch-1:
            print("data shape:",data.shape)
            num_frames = data.shape[1]
            print("number of frames in last batch:",num_frames)
            zeros = torch.zeros((3,batch_size-num_frames,224,224),dtype=torch.uint8)
            print("Zeros size:",zeros.shape)
            data = torch.cat((data,zeros),axis=1)
            print("Data size last batch:",data.shape)
        data = data.unsqueeze(0)
        concat_emb = torch.cat((concat_emb,data))
        if count % 16 == 0 or count == num_batch:
            print(i)
            with torch.no_grad():
                print("Concatenated embeddings:",concat_emb.shape)
                embedding = net(concat_emb.to(device))
                concat_emb = torch.tensor([]).cpu()
            emb = torch.cat((emb,embedding["video_embedding"]))
    if not os.path.exists(opj("raw_frames", dest)):
        os.makedirs(opj("raw_frames", dest))

    new_path = opj("raw_frames", dest)
    print("Final embeddings shape:",emb.shape)
    torch.save(emb, opj(new_path, ''.join((vid.split('.mp4')[0], '.pth'))))
    if not os.path.exists(opj("raw_frames", dest)):
        os.makedirs(opj("raw_frames", dest))

    new_path = opj("raw_frames", dest)
    torch.save(emb, opj(new_path, ''.join((vid.split('.mp4')[0], '.pth'))))

if __name__ == '__main__':
    all_files = glob.glob('/common/home/apc120/video/*/*.mp4')
    completed = []
    if os.path.exists('done_test.txt'):
        with open('done_test.txt', 'r') as f:
            completed = f.readlines()

    completed = [i.strip() for i in completed]

    for i in tqdm(all_files):
        print(i)
        if i in completed:
            print('skipping', i)
            continue
        extract_and_save(i)
        with open('done_test.txt', 'a') as f:
            f.write(i + '\n')    
        


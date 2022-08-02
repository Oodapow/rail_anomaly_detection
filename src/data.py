from matplotlib.pyplot import axis
import torch
import os
import numpy as np
import cv2
import json

LABELS = ['road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign', 'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded']

def make_collate_fn(in_dim=(1920, 1080), out_dim=(480, 270)):
    def collate_fn(batch):
        images = torch.stack([torch.tensor(cv2.resize(b[0].astype('float32'), in_dim)).permute(2, 0, 1).div(255.) for b in batch])
        outputs = torch.stack([torch.tensor(cv2.resize(b[1].astype('float32'), out_dim)).permute(2, 0, 1).div(255.) for b in batch])
        output_images = torch.stack([torch.tensor(cv2.resize(b[0].astype('float32'), out_dim)).permute(2, 0, 1).div(255.) for b in batch])

        return images, outputs, output_images
    return collate_fn

class RailSemDataset(torch.utils.data.Dataset):
    def __init__(self, path, ids):
        super().__init__()
        self.path = path
        self.ids = list(ids)
        with open(os.path.join(path, 'rs19-config.json'), 'r') as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        input_path = os.path.join(self.path, 'jpgs', 'rs19_val', f'rs{id:05}.jpg')
        output_path = os.path.join(self.path, 'uint8', 'rs19_val', f'rs{id:05}.png')

        input_tensor = cv2.imread(input_path)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        
        im_id_map = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        outputs = []
        for i, _ in enumerate(self.config['labels']):
            img = ((im_id_map == i).astype(int) * 255)[:,:,np.newaxis]
            outputs.append(img)
        
        output_tensor = np.concatenate(outputs, axis=2)

        return input_tensor, output_tensor
        
if __name__ == '__main__':

    ds = RailSemDataset('/home/oodapow/data/rs19', range(10))

    print('len:', len(ds))

    it, ot = ds[0]


    print('it shape:', it.shape)
    print('ot shape:', ot.shape)

    collate_fn = make_collate_fn()

    it, ot, iot = collate_fn([(it, ot)])

    print('collate_fn it shape:', it.shape)
    print('collate_fn ot shape:', ot.shape)
    print('collate_fn iot shape:', iot.shape)

    labels = []


    # (2, 0, 1)
    for i,l in enumerate(ds.config['labels']):
        labels.append(l['name'])
        cm = ot[:,i,:,:].unsqueeze(0)
        im = iot * cm * 255.
        im = im[0]
        im = im.permute((1, 2, 0)).numpy()
        print(im.shape)
        cv2.imwrite(f'{l["name"]}.png', im)

    print(labels)
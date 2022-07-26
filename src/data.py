from matplotlib.pyplot import axis
import torch
import os
import numpy as np
import cv2
import json

LABELS = ['road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign', 'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded']

def make_collate_fn(augumenter=lambda x: x):
    def collate_fn(batch):
        images = torch.stack([torch.tensor(b[0]).permute(2, 0, 1).div(255.) for b in batch])
        outputs = torch.stack([torch.tensor(b[1]).permute(2, 0, 1).div(255.) for b in batch])

        tensor = torch.cat([images, outputs], axis=1)
    
        tensor = augumenter(tensor)

        return tensor[:, 0:3, :, :], tensor[:, 3:, :, :]
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

    labels = []

    for i,l in enumerate(ds.config['labels']):
        labels.append(l['name'])
        cv2.imwrite(f'{l["name"]}.png', it * ot[:,:,i][:,:,np.newaxis] / 255.)

    print(labels)

    collate_fn = make_collate_fn()

    it, ot = collate_fn([(it, ot)])

    print('collate_fn it shape:', it.shape)
    print('collate_fn ot shape:', ot.shape)
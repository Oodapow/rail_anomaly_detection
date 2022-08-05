import torch
import os
import numpy as np
import cv2
import json

LABELS =      ['road', 'sidewalk', 'construction', 'tram-track', 'fence', 'pole', 'traffic-light', 'traffic-sign', 'vegetation', 'terrain', 'sky', 'human', 'rail-track', 'car', 'truck', 'trackbed', 'on-rails', 'rail-raised', 'rail-embedded']

LABELS_DIST = [0.052,   0.031,      0.122,          0.031,        0.025,   0.028,  0.003,           0.002,          0.233,        0.066,     0.223, 0.005,   0.059,        0.008, 0.007,   0.103,      0.033,      0.034,         0.015,         0.054]

LABELS_WEIGHTS = [ 1 / l / len(LABELS_DIST) for l in LABELS_DIST ]

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    outputs = torch.stack([b[2] for b in batch])
    output_images = torch.stack([b[3] for b in batch])
    return images, targets, outputs, output_images

class RailSemDataset(torch.utils.data.Dataset):
    def __init__(self, path, ids, in_dim=(960, 540), out_dim=(960, 540)):
        super().__init__()
        self.path = path
        self.ids = list(ids)
        self.in_dim = in_dim
        self.out_dim = out_dim
        with open(os.path.join(path, 'rs19-config.json'), 'r') as f:
            self.config = json.load(f)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        id = self.ids[index]

        cache_path = os.path.join(self.path, 'cache', 'rs19_val', f'rs{id:05}_{self.in_dim[0]}x{self.in_dim[1]}_{self.out_dim[0]}x{self.out_dim[1]}.pt')

        if os.path.isfile(cache_path):
            return torch.load(cache_path)

        input_path = os.path.join(self.path, 'jpgs', 'rs19_val', f'rs{id:05}.jpg')
        output_path = os.path.join(self.path, 'uint8', 'rs19_val', f'rs{id:05}.png')

        input_tensor = cv2.imread(input_path)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        
        im_id_map = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)

        outputs = []
        for i, _ in enumerate(LABELS):
            img = ((im_id_map == i).astype(int) * 255)[:,:,np.newaxis]
            outputs.append(img)
        
        output_tensor = np.concatenate(outputs, axis=2)

        target_tensor = torch.tensor(cv2.resize(im_id_map.astype('float32'), self.out_dim, interpolation=cv2.INTER_NEAREST), dtype=torch.long).permute(0, 1)

        res = (
            torch.tensor(cv2.resize(input_tensor.astype('float32'), self.in_dim)).permute(2, 0, 1).div(255.), 
            torch.where(target_tensor == 255, len(LABELS), target_tensor),
            torch.ceil(torch.tensor(cv2.resize(output_tensor.astype('float32'), self.out_dim)).permute(2, 0, 1).div(255.)),
            torch.tensor(cv2.resize(input_tensor.astype('float32'), self.out_dim)).permute(2, 0, 1).div(255.),
        )

        torch.save(res, cache_path)

        return res
        
if __name__ == '__main__':

    ds = RailSemDataset('/home/oodapow/data/rs19', range(10))

    print('len:', len(ds))

    it, tgt, ot, iot = ds[0]


    print('it shape:', it.shape)
    print('tgt shape:', tgt.shape)
    print('ot shape:', ot.shape)
    print('iot shape:', iot.shape)

    it, tgt, ot, iot = collate_fn([(it, tgt, ot, iot)])

    print('collate_fn it shape:', it.shape)
    print('collate_fn tgt shape:', tgt.shape)
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
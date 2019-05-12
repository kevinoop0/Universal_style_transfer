import os
from torch.utils.data import Dataset
import PIL
import numpy as np
from PIL import Image
import torchvision.transforms.functional as transforms

def load_img(path, new_size):
    img = Image.open(path).convert(mode='RGB')
    if new_size:
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
    return transforms.to_tensor(img)

class ContentStylePairDataset(Dataset):

    def __init__(self, args):
        super(Dataset, self).__init__()
        self.contentSize = args.contentSize
        self.styleSize = args.styleSize
        self.pairs_fn = [(args.content, args.style)]


    def __len__(self):
        return len(self.pairs_fn)

    def __getitem__(self, idx):
        pair = self.pairs_fn[idx]
        style = load_img(pair[1], self.styleSize)
        content = load_img(pair[0], self.contentSize)
        return {'content': content, 'contentPath': pair[0], 'style': style, 'stylePath': pair[1]}

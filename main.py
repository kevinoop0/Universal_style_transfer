# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torchvision
import autoencoder
import PairDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform')
    parser.add_argument('--content', help='Path of the content image (or a directory containing images) to be trasformed')
    parser.add_argument('--style', help='Path of the style image (or a directory containing images) to use')
    parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
    parser.add_argument('--outDir', default='outputs', help='Path of the directory where stylized results will be saved')
    parser.add_argument('--no-cuda', default=False, action='store_true', help='Flag to enables GPU (CUDA) accelerated computations')
    parser.add_argument('--alpha', type=float, default=0.2, help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyperparameter balancing the interpolation between the two images in the stylePair')
    parser.add_argument('--contentSize', type=int, default=512, help='Reshape content image to have the new specified maximum size (keeping aspect ratio)') # default=768 in the paper
    parser.add_argument('--styleSize', type=int, default=512, help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

    return parser.parse_args()

def save_image(img, content_name, style_name, out_ext, args):
    torchvision.utils.save_image(img.cpu().detach().squeeze(0),
     os.path.join(args.outDir, ''+ content_name + '_stylized_by_' + style_name + '_alpha_' + str(int(args.alpha*100)) + '.' + out_ext))


def main():
    args = parse_args()
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    model = autoencoder.MultiLevelWCT(args)
    model.to(device=args.device)
    model.eval()
    dataset = PairDataset.ContentStylePairDataset(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        s_basename = str(os.path.basename(sample['stylePath'][0]).split('.')[0])
        content = sample['content'].to(device=args.device)
        style = sample['style'].to(device=args.device)
        c_basename = str(os.path.basename(sample['contentPath'][0]).split('.')[0])
        c_ext = str(os.path.basename(sample['contentPath'][0]).split('.')[-1])
        out = model(content, style)
        save_image(out, c_basename, s_basename, c_ext, args)

if __name__ == "__main__":
    main()


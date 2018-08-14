from collections import namedtuple
import scipy as sp
import numpy as np
import cv2
import argparse
from monodepth_dataloader import *

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')

parser.add_argument('--mode',                      type=str,   help='train, test, or single', default='train')
parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default='utils/filenames/kitti_stereo_2015_test_files.txt')#required=True)
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')

parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--input_height',              type=int,   help='input height', default=192)
parser.add_argument('--input_width',               type=int,   help='input width', default=384)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=100)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.0)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='log')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')


args = parser.parse_args()

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')

params = monodepth_parameters(
    encoder=args.encoder,
    height=args.input_height,
    width=args.input_width,
    batch_size=args.batch_size,
    num_threads=args.num_threads,
    num_epochs=args.num_epochs,
    do_stereo=args.do_stereo,
    wrap_mode=args.wrap_mode,
    use_deconv=args.use_deconv,
    alpha_image_loss=args.alpha_image_loss,
    disp_gradient_loss_weight=args.disp_gradient_loss_weight,
    lr_loss_weight=args.lr_loss_weight,
    full_summary=args.full_summary)   


if __name__ == '__main__':
    pred_right = np.load(args.predicted_disp_path)

    if args.split == 'kitti':
        num_samples = 200
        dataloader = MonodepthDataloader(args.gt_path, args.filenames_file, params, args.dataset, args.mode)
        gt_rights = dataloader.right_image_batch

    mae = np.zeros(num_samples, np.float32) 
    
    _, ori_height, ori_weight, _ = pred_right.shape

    for i in range(num_samples):
        print('gt_rights[i] shape: ', gt_rights[i].shape)
        gt_right = sp.misc.imresize(gt_rights[i].astype(np.uint8), [ori_height, ori_weight])
        pred_right = pred_rights[i]

        if args.split == 'kitti':
            mae[i] = np.round(np.abs(gt_right[i] - pred_right[i]).mean(),2)
    print('mae: ', np.round(mae.mean(),2))


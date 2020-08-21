import argparse
import logging
import os
import torch
import numpy as np

from engine.ml_with_loss_engine import MLWithLossEngine
import models.res_selective_fc as res
import models.vgg_selective_fc as vgg
from voc import Voc2007Classification

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', type=str,
                    default='',)
parser.add_argument('--model_path_prefix', type=str, 
                    default='')
parser.add_argument('--model_name', type=str, 
                    default='')
parser.add_argument('--net_name', type=str, 
                    default='res101')
                    # default='vgg19bn')
                    # default='mbv2')
                    # default='res18')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default='[30]', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float,)

parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--re_probability', default=0.5, type=float,)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--logfile', default='', type=str)
parser.add_argument('--only_hard', default=1, type=int)


name2net = {
    'res152': res.resnet152, # 3205M
    'res101': res.resnet101, # 2619M
    'res50': res.resnet50,
    'res18': res.resnet18,
    'res32x8': res.resnext101_32x8d,  # 4277M
    'res32x16': res.resnext101_32x16d,  # 7127M
    # 'res32x32': res.resnext101_32x32d, 
    'mbv2': res.mobilenet_v2,
    'vgg16': vgg.vgg16,  # 3675M
    'vgg16bn': vgg.vgg16bn,  # 3675M
    'vgg19bn': vgg.vgg19bn,  # 4223M
}


def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.logfile:
        handlers.append(logging.FileHandler(args.logfile, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-5s %(message)s',
        datefmt='%m-%d %H:%M:%S', handlers=handlers,
    )

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval')
    val_hard_only_dataset = Voc2007Classification(args.data, 'test', only_hard=args.only_hard)
    val_dataset = Voc2007Classification(args.data, 'test')

    num_classes = 20

    criterion = torch.nn.BCEWithLogitsLoss()
    # load model
    model = name2net[args.net_name](
        num_classes = num_classes, alpha = args.alpha, dropout=args.dropout
    )

    if not args.model_name:
        all_model_name = [x for x in os.listdir(args.model_path_prefix) if '.pth' in x ]
    else:
        all_model_name = [args.model_name]
    val_results_list = []
    val_hard_results_list = []

    for model_name in all_model_name:
        logging.info(f'loading {model_name}')
        state_dict = torch.load(os.path.join(args.model_path_prefix, model_name))['state_dict']
        if 's_net' in ''.join(state_dict.keys()):
            new = {}
            for k, v in state_dict.items():
                if 's_net' in k:
                    new[k.replace('s_net.', '')] = v
            state_dict = new
        model.load_state_dict(state_dict)

        # define optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )


        state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
                'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
        state['difficult_examples'] = True
        state['save_model_path'] = './data/out/'
        state['workers'] = args.workers
        state['epoch_step'] = eval(args.epoch_step)
        state['lr'] = args.lr
        state['min_lr'] = args.min_lr
        state['print_freq'] = args.print_freq
        state['re_probability'] = args.re_probability
        state['evaluate'] = True
        engine = MLWithLossEngine(state)
        val_results = engine.learning(model, criterion, train_dataset, val_dataset, optimizer)
        val_hard_results = engine.learning(model, criterion, train_dataset, val_hard_only_dataset, optimizer)
        val_results_list.append(val_results.item())
        val_hard_results_list.append(val_hard_results.item())
    val_results_list = np.asarray(val_results_list)
    val_hard_results_list = np.asarray(val_hard_results_list)
    logging.info(f'val mean {val_results_list.mean()} std {val_results_list.std()}')
    logging.info(f'val hard mean {val_hard_results_list.mean()} std {val_hard_results_list.std()}')



if __name__ == '__main__':
    main_voc2007()

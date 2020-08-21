import argparse

import torch

from engine import MultiLabelMAPEngine
from models.wildcat import resnet101_wildcat
from voc import Voc2007Classification

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', type=str,
                    default='/home/users/liangchen.song/data/seg/voc',)
parser.add_argument('--model_path_prefix', type=str, 
                    default='/home/users/liangchen.song/data/models')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default='[30]', type=str)
parser.add_argument('--warm_up_epoch', default=10, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--re_probability', default=0.5, type=float,)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval')
    val_dataset = Voc2007Classification(args.data, 'test')

    num_classes = 20

    # load model
    model = resnet101_wildcat(num_classes, args.model_path_prefix, kmax=0.2, alpha=0.7, num_maps=8)

    # define loss function (criterion)
    # criterion = torch.nn.MultiLabelSoftMarginLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
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
    state['print_freq'] = args.print_freq
    state['warm_up_epoch'] = args.warm_up_epoch
    state['re_probability'] = args.re_probability
    if args.evaluate:
        state['evaluate'] = True
    engine = MultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    main_voc2007()

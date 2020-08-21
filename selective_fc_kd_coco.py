import argparse
import logging
import torch

from engine.ml_with_loss_engine import MLWithLossEngine
import models.res_selective_fc as res
import models.vgg_selective_fc as vgg
from models.selective_fc_kd import SelectiveFCKD
from coco import COCO2014

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--data', type=str,
                    default='/home/users/liangchen.song/data/det/coco2014',)
parser.add_argument('--model_path_prefix', type=str, 
                    default='/home/users/liangchen.song/data/models')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default='[30]', type=str)
parser.add_argument('--warm_up_epoch', default=0, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float,)
parser.add_argument('--dropout', default=0.2, type=float,)

parser.add_argument('--temperature', default=3, type=float,)
parser.add_argument('--alpha', default=1, type=float,)
parser.add_argument('--cls_balance', default=0, type=float,)

parser.add_argument('--re_probability', default=0.5, type=float,)
parser.add_argument('--restart_times', default=4, type=int,)
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

parser.add_argument('--t_net', default='vgg19bn', type=str)
parser.add_argument('--t_net_model', type=str,
                    default='/mnt/data-1/liangchen.song/data/models/trained/distill-ml/vgg19bn_coco_model_best_77.7904.pth.tar')
parser.add_argument('--s_net', default='res18', type=str)
parser.add_argument('--logfile', default='', type=str)


name2net = {
    'res152': res.resnet152,  # 3205M
    'res101': res.resnet101,  # 2619M
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


def main_coco():
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
    train_dataset = COCO2014(args.data, phase='train')
    val_dataset = COCO2014(args.data, phase='val')

    num_classes = 80

    # load model
    t_net = name2net[args.t_net](num_classes=num_classes)
    t_net.load_state_dict(torch.load(args.t_net_model)['state_dict'])
    s_net = name2net[args.s_net](args.model_path_prefix, num_classes=num_classes, dropout=args.dropout)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    distill_model = SelectiveFCKD(
        s_net, t_net, criterion, cls_balance=args.cls_balance,
        alpha=args.alpha, temperature=args.temperature
    )

    # define optimizer
    optimizer = torch.optim.SGD(
        s_net.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    state = {
        'batch_size': args.batch_size, 'image_size': args.image_size, 
        'max_epochs': args.epochs, 'evaluate': args.evaluate, 
        'resume': args.resume, 'num_classes':num_classes,
    }
    state['difficult_examples'] = True
    state['save_model_path'] = './data/out/'
    state['workers'] = args.workers
    state['epoch_step'] = eval(args.epoch_step)
    state['lr'] = args.lr
    state['min_lr'] = args.min_lr
    state['print_freq'] = args.print_freq
    state['warm_up_epoch'] = args.warm_up_epoch
    state['re_probability'] = args.re_probability
    state['restart_times'] = args.restart_times
    state['alpha'] = args.alpha
    if args.evaluate:
        state['evaluate'] = True
    engine = MLWithLossEngine(state)

    # engine.init_learning(distill_model)
    # val_dataset.transform = engine.state['val_transform']
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers
    # )
    # class T(torch.nn.Module):
    #     def __init__(self, t=t_net):
    #         super().__init__()
    #         self.t = t
    #     def forward(self, x, y):
    #         return self.t(x, y)[0], torch.zeros([1,1]).to(x.device)
    # t = T()
    # print('Validating Teacher Net:')
    # engine.validate(val_loader, torch.nn.DataParallel(t).cuda(), criterion)

    engine.learning(distill_model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    main_coco()

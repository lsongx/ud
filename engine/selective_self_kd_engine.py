import math
import logging
import os
import torch

from .util import AveragePrecisionMeter
from .multi_label_map_engine import MultiLabelMAPEngine
from .engine import Engine


class SelectiveSelfKDEngine(MultiLabelMAPEngine):
    def __init__(self, state):
        super().__init__(state)

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        super().on_start_batch(training, model, criterion, data_loader, optimizer, display)
        if optimizer is not None:
            epochs = self.state['max_epochs'] / self.state['restart_times']
            current_epoch = self.state['epoch'] % epochs
            current_iter = self.state['iteration'] + current_epoch * len(data_loader)
            total_iters = epochs * len(data_loader) - 1
            lr = self.state['min_lr'] + 0.5 * (self.state['lr'] - self.state['min_lr']) * (1+math.cos(math.pi*(current_iter/total_iters)))
            self.state['current_lr'] = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if lr == self.state['min_lr']:
                self.state['copy_teacher'] = True
                logging.info('Start/updating the teacher net.')
            else:
                self.state['copy_teacher'] = False

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = self.state['input']
        target_var = self.state['target']

        # compute output
        if not training:
            with torch.no_grad():
                self.state['output'] = model(input_var, 0)[0]
        else:
            self.state['output'], loss = model(input_var, target_var)
            self.state['loss'] = loss.mean()

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        if self.state['copy_teacher']:
            load_dict = torch.load(os.path.join(
                self.state['save_model_path'], 'model_best.pth.tar'
            ))
            new_dict = {}
            for k, v in load_dict['state_dict'].items():
                if 's_net' in k:
                    new_dict[k.replace('s_net.', '')] = v
            model.module.t_net.load_state_dict(new_dict)
            model.module.cls_balance = self.state['cls_balance']
            model.module.alpha = self.state['alpha']
            self.state['lr'] = self.state['kd_lr']

    def adjust_learning_rate(self, optimizer):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'
                      'LR {lr:.2e}'.format(
                          self.state['epoch'], self.state['iteration'], len(
                              data_loader),
                          batch_time_current=self.state['batch_time_current'],
                          batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                          data_time=data_time, loss_current=self.state['loss_batch'], loss=loss, lr=self.state['current_lr']))
            else:
                logging.info('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

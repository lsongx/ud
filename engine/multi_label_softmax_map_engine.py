import math

from .util import AveragePrecisionMeter
from .engine import Engine

import torch


class MultiLabelSoftmaxMAPEngine(Engine):
    def __init__(self, state):
        super().__init__(state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        mAP = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print(f'Epoch: [{self.state["epoch"]}]\t'
                      f'Loss {loss:.4f}\t'
                      f'mAP {mAP:.3f}')
                print(f'OP: {OP:.4f}\t'
                      f'OR: {OR:.4f}\t'
                      f'OF1: {OF1:.4f}\t'
                      f'CP: {CP:.4f}\t'
                      f'CR: {CR:.4f}\t'
                      f'CF1: {CF1:.4f}')
            else:
                print(f'Test: \t Loss {loss:.4f}\t mAP {mAP:.3f}')
                print(f'OP: {OP:.4f}\t'
                      f'OR: {OR:.4f}\t'
                      f'OF1: {OF1:.4f}\t'
                      f'CP: {CP:.4f}\t'
                      f'CR: {CR:.4f}\t'
                      f'CF1: {CF1:.4f}')
                print(f'OP_3: {OP_k:.4f}\t'
                      f'OR_3: {OR_k:.4f}\t'
                      f'OF1_3: {OF1_k:.4f}\t'
                      f'CP_3: {CP_k:.4f}\t'
                      f'CR_3: {CR_k:.4f}\t'
                      f'CF1_3: {CF1_k:.4f}')

        return mAP

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

        if optimizer is not None:
            current_iter = self.state['iteration'] + self.state['epoch'] * len(data_loader)
            # warm_up_iters = self.state['warm_up_epoch'] * len(data_loader)
            # if warm_up_iters > 0:
            #     k = (self.state['lr'] - 5e-5) / warm_up_iters
            #     if current_iter < warm_up_iters:
            #         lr = k * current_iter + 5e-5
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = lr
            total_iters = self.state['max_epochs'] * len(data_loader)
            lr = 1e-4 + 0.5 * (self.state['lr'] - 1e-4) * (1+math.cos(math.pi * (current_iter/total_iters)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def adjust_learning_rate(self, optimizer):
        pass

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = self.state['input']
        target_var = self.state['target'].long()

        # compute output
        if not training:
            with torch.no_grad():
                self.state['output'] = model(input_var)
        else:
            self.state['output'] = model(input_var)
        self.state['loss'] = criterion(self.state['output'], target_var)
        self.state['output'] = self.state['output'].softmax(dim=1)[:,1,:]

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

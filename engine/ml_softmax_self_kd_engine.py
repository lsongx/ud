import math
import logging
import torch

from .util import AveragePrecisionMeter
from .multi_label_map_engine import MultiLabelMAPEngine


class MLSoftmaxSelfKDEngine(MultiLabelMAPEngine):
    def __init__(self, state):
        super().__init__(state)

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        super().on_start_batch(training, model, criterion, data_loader, optimizer, display)
        if optimizer is not None:
            current_iter = self.state['iteration'] + self.state['epoch'] * len(data_loader)
            total_iters = self.state['max_epochs'] * len(data_loader) / self.state['restart_times']
            lr = 1e-4 + 0.5 * (self.state['lr'] - 1e-4) * (1+math.cos(math.pi * (current_iter/total_iters)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if lr == 1e-4:
                self.state['copy_teacher'] = True
                logging.info('Start/updating the teacher net.')
            else:
                self.state['copy_teacher'] = False


    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = self.state['input']
        target_var = self.state['target']

        if self.state['copy_teacher']:
            model.module.t_net.load_state_dict(model.module.s_net.state_dict())
            model.module.cls_balance = self.state['cls_balance']

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
            optimizer.step()

    def adjust_learning_rate(self, optimizer):
        pass

import math
import torch

from .util import AveragePrecisionMeter
from .multi_label_map_engine import MultiLabelMAPEngine


class MLWithLossEngine(MultiLabelMAPEngine):
    def __init__(self, state):
        super().__init__(state)

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        super().on_start_batch(training, model, criterion, data_loader, optimizer, display)
        if optimizer is not None:
            current_iter = self.state['iteration'] + self.state['epoch'] * len(data_loader)
            total_iters = self.state['max_epochs'] * len(data_loader)
            lr = self.state['min_lr'] + 0.5 * (self.state['lr'] - self.state['min_lr']) *\
                (1+math.cos(math.pi * (current_iter/total_iters)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = self.state['input']
        target_var = self.state['target']

        # compute output
        if not training:
            with torch.no_grad():
                self.state['output'] = model(input_var, 0)[0]
                self.state['loss'] = torch.zeros(1)
        else:
            # import pdb; pdb.set_trace()
            # model.module.s_net.pooling_bce.weight.grad.sum()
            # self.state['output'], loss = model(input_var, target_var, f_type=2)
            self.state['output'], loss = model(input_var, target_var)
            # .mean() here for multi gpu training
            self.state['loss'] = loss.mean()
            optimizer.zero_grad()
            self.state['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

    def adjust_learning_rate(self, optimizer):
        pass

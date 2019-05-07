from typing import Optional, List, Union, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(
            self,
            model: torch.Tensor,
            loss: torch.Tensor,
            metrics: List[torch.Tensor],
            optimizer: torch.Tensor,
            resume: Union[str, Path],
            config: Dict,
            data_loader: DataLoader,
            valid_data_loader: Optional[DataLoader]=None,
            train_logger=None
    ):
        '''
        trainer class

        :param model: model class
        :param loss: loss to train
        :param metrics: list of metrics to be used
        :param optimizer: optimizer
        :param config: configuration parameters
        :param data_loader: data loader
        :param valid_data_loader: validation data loader
        :param resume: path to ch eckpoint to resume training
        :param train_logger: logger for training
        '''
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output: torch.Tensor, target: torch.Tensor):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch: int):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss, total_ppl = 0, 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (src, tgt, lengths) in enumerate(self.data_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)

            self.model.zero_grad()
            output = self.model(src, tgt[:-1], lengths)  # exclude last target from inputs
            loss = self.loss(output, tgt[1:].view(-1))  # exclude <SOS> from targets
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalar('ppl', np.exp(loss.item()))
            total_loss += loss.item()
            total_ppl += np.exp(loss.item())
            total_metrics += self._eval_metrics(output, tgt[1:].view(-1))

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} PPL: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item(),
                    np.exp(loss.item()),
                ))

        log = {
            'loss': total_loss / len(self.data_loader),
            'ppl': total_ppl / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        self.optimizer.update_learning_rate(log['val_loss'], epoch)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss, total_val_ppl = 0, 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (src, tgt, lengths) in enumerate(self.data_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)

                output = self.model(src, tgt[:-1], lengths)  # exclude last target from inputs
                loss = self.loss(output, tgt[1:].view(-1))  # exclude <SOS> from targets

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_ppl += np.exp(loss.item())
                total_val_metrics += self._eval_metrics(output, tgt[1:].view(-1))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_ppl': total_val_ppl / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

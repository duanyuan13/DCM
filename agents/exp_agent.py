"""
Capsule agent
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')

import time
import shutil
import logging
import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from os.path import join
from tqdm import tqdm
from itertools import compress

from agents.base_agent import BaseAgent
from dataset.utils.get_datasets import get_datasets
from models.utils.get_model import get_model
from losses.utils.get_losses import get_loss_module
from optimizers.utils.get_lr_schedule import get_lr_schedule
from optimizers.utils.get_optimizer import get_optimizer
from metrics.average_meter import AverageMeter
from metrics.compute_metrics import compute_abs_rel_error, compute_square_rel_error

# set True for consistence input size only
torch.backends.cudnn.benchmark = True


class ExpAgent(BaseAgent):
    """
    Agent

    Args:
        config (config node object): the given config for the agent
    """

    def __init__(self, config):
        super(ExpAgent, self).__init__(config)

        self.logger = logging.getLogger('Exp Agent')

        self.summ_writer = SummaryWriter(log_dir=self.config.env.summ_dir,
                                         comment='Exp Agent')

        self.is_cuda_available = torch.cuda.is_available()
        self.use_cuda = self.is_cuda_available and self.config.env.use_cuda

        if self.use_cuda:
            self.device = torch.device('cuda:' + self.config.env.cuda_id)
            self.logger.info('Agent running on CUDA')
            torch.manual_seed(self.config.env.seed)
            torch.cuda.manual_seed_all(self.config.env.seed)
            np.random.seed(self.config.env.seed)
        else:
            self.device = torch.device('cpu')
            self.logger.info('Agent running on CPU')
            torch.manual_seed(self.config.env.seed)

        self.train_set = get_datasets(self.config.data,
                                      self.config.data.train.name)

        self.valid_set = get_datasets(self.config.data,
                                      self.config.data.val.name)
        self.test_set = get_datasets(self.config.data,
                                     self.config.data.test.name)
        self.logger.info('processing: get datasets')

        self.train_loader = DataLoader(dataset=self.train_set,
                              batch_size=self.config.data.batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)

        self.valid_loader = DataLoader(dataset=self.valid_set,
                              batch_size=self.config.data.valid_batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)

        self.test_loader = DataLoader(dataset=self.test_set,
                              batch_size=self.config.data.test_batch_size,
                              shuffle=self.config.data.shuffle, sampler=None,
                              batch_sampler=None, num_workers=0,
                              pin_memory=self.config.data.pin_memory,
                              drop_last=self.config.data.drop_last)
        
        self.logger.info('processing: dataset loader')

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_metric = 1

        self.offset_model = get_model(self.config.model.offset)
        self.offset_model = self.offset_model.to(self.device)

        self.dist_model = get_model(self.config.model.dist)
        self.dist_model = self.dist_model.to(self.device)

        self.loss = get_loss_module(self.config)
        self.logger.info('processing: get loss module')
        self.loss = self.loss.to(self.device)

        self.optimizer = get_optimizer(self.config,
                                       self.offset_model.parameters())

        self.scheduler = get_lr_schedule(self.config, self.optimizer)

        # try to load existing ckpt to resume a interrupted training
        self.resume(self.config.ckpt.ckpt_name)

    def resume(self, ckpt_name='latest_ckpt.pth'):
        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        try:
            self.load_ckpt(ckpt_path)
        except:
            self.logger.info('Can not load ckpt at "%s"', ckpt_path)

    def load_ckpt(self, ckpt_path, strict=False):
        """
        Load checkpoint with given ckpt_name

        Args:
            ckpt_path (string): the path to ckpt
            strict (bool): whether or not to strictly load ckpt
        """

        try:
            self.logger.info('Loading ckpt from %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.device)

            self.current_epoch = ckpt['current_epoch']
            self.current_iteration = ckpt['current_iteration']

            self.offset_model.load_state_dict(ckpt['model_state_dict'], strict=strict)
            # NOTE
            # self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            self.logger.info('Successfully loaded ckpt from %s at '
                             'epoch %d, iteration %d', ckpt_path,
                             self.current_epoch, self.current_iteration)
            self.logger.info(
                'Loaded initial learning rate %f from ckpt',
                ckpt['optimizer_state_dict']['param_groups'][0]['lr']
            )

        except OSError:
            self.logger.warning('No ckpt exists at "%s". Skipping...',
                                ckpt_path)

    def save_ckpt(self, ckpt_name='ckpt.pth', is_best=False):
        """
        Save the current state_dict of agent model to ckpt_path

        Args:
            ckpt_name (string, optional): the name of the current state_dict to
                 save as
            is_best (bool, optional): indicator for whether the model is best
        """
        state = {'current_epoch': self.current_epoch,
                 'current_iteration': self.current_iteration,
                 'model_state_dict': self.offset_model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict()}

        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)

        if is_best:
            best_ckpt_path = join(self.config.env.ckpt_dir,
                                  'best_' + ckpt_name)
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def run(self):
        """
        The main operator of agent
        """
        try:
            if self.config.agent.mode == 'valid':
                self.validate()

            elif self.config.agent.mode == 'train':
                self.train()

            else:
                self.logger.error('Running mode %s not implemented',
                                  self.config.agent.mode)
                raise NotImplementedError

        except KeyboardInterrupt:
            self.logger.info('Agent interrupted by CTRL+C...')

    def train(self):
        """
        Main training loop
        """
        self.val_iter = 0
        self.test_iter = 0
        for i in range(self.current_epoch, self.config.optimizer.max_epoch):
            # train one epoch
            self.train_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            else:
                self.logger.info('lr scheduler is None')

            if i % self.config.ckpt.save_interval == 0:
                valid_epoch_loss, abs_rel = self.validate()
                is_best = abs_rel < self.best_valid_metric

                if is_best:
                    self.best_valid_metric = abs_rel
                    self.save_ckpt(is_best=is_best)
                    """
                    test only model has a lowest abs rel on val set
                    """
                _, _ = self.validate(mode='test')

                ckpt_name = 'ckpt_' + str(self.current_epoch) + '.pth'
                self.save_ckpt(ckpt_name)
            self.current_epoch += 1

    def base_forward(self, inputs_xywh):
        B = inputs_xywh.size()[0]

        # inputs_xx shape: [B, 2, 1]
        # every batch contains one left x and one right x
        inputs_xx = inputs_xywh[:, :, 0:1]
        inputs_xx = torch.reshape(inputs_xx, (B, 2))

        # inputs_adwh: inputs containing: angle, distance(to image center), width, height
        inputs_adwh = self.train_set.cal_rotate_dist(inputs_xywh, is_norm=self.config.data.norm_rad)  # re [2*B, 4]

        inputs_adwh = inputs_adwh.reshape(2 * B, 1, -1, 1)  # [2*B, 1, 4, 1]

        # forward propagation
        # offset_preds shape: [2*B, 4, 1]
        offset_preds = self.offset_model(inputs_adwh)  # return [2*B, 1]

        # offset_preds shape: [B, 2]
        offset_preds = torch.reshape(offset_preds, (B, -1))  # return [B, 2]
        distance = self.dist_model(offset_preds, inputs_xx)

        return distance

    def train_one_epoch(self):
        """
        One epoch of training
        """
        if not self.config.data.drop_last:
            iteration_per_epoch = int(np.ceil(len(self.train_set) /
                                              self.train_loader.batch_size))
        else:
            iteration_per_epoch = int(len(self.train_set) /
                                      self.train_loader.batch_size)

        # init train batch
        tqdm_batch = tqdm(iterable=self.train_loader,
                          total=iteration_per_epoch,
                          desc='Train epoch {}'.format(self.current_epoch))

        # set model into train mode
        self.offset_model.train()
        self.dist_model.train()

        # initialize average meters
        epoch_loss = AverageMeter()
        epoch_abs = AverageMeter()
        epoch_sq = AverageMeter()

        # stare the training loop over iterations of one epoch
        for inputs, gt_distance, file_id in tqdm_batch:
            # inputs shape: [B, 2, 4]
            inputs = inputs.to(self.device)
            gt_distance = gt_distance.to(self.device)

            distance = self.base_forward(inputs)

            # loss function
            curr_loss = self.loss(distance, gt_distance)
            # optimizer
            self.optimizer.zero_grad()
            curr_loss.backward()

            if self.config.optimizer.clip_grad.is_clip:
                torch.nn.utils.clip_grad_norm_(self.offset_model.parameters(),
                                               self.config.optimizer.clip_grad.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(curr_loss):
                self.logger.error('Loss is NaN during training...')
                raise RuntimeError

            batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
            batched_sq_rel = compute_square_rel_error(distance, gt_distance)

            # update average meter
            epoch_loss.update(curr_loss.item())
            for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                epoch_abs.update(abs_rel)
                epoch_sq.update(sq_rel)

            batched_abs_std = batched_abs_rel.std()
            self.summ_writer.add_scalar('train/abs_std',
                                        batched_abs_std,
                                        self.current_iteration, time.time())

            self.current_iteration += 1

        tqdm_batch.close()

        self._logger(key='train',
                     epoch_dist_loss=epoch_loss,
                     epoch_abs=epoch_abs,
                     epoch_sq=epoch_sq)

        self.logger.info('NUM: %d | train epoch: %d | lr: %f | loss: %8f | ABS: %f | SQ: %f',
                         len(self.train_set),
                         self.current_epoch,
                         self.optimizer.param_groups[0]['lr'],
                         epoch_loss.val,
                         epoch_abs.val,
                         epoch_sq.val)

    def _logger(self,
                key,
                epoch_dist_loss,
                epoch_abs,
                epoch_sq):
        self.summ_writer.add_scalar(f'{key}/dist_loss',
                                    epoch_dist_loss.val,
                                    self.current_epoch, time.time())

        self.summ_writer.add_scalar(f'{key}/abs',
                                    epoch_abs.val,
                                    self.current_epoch,
                                    time.time())

        self.summ_writer.add_scalar(f'{key}/learning_rate',
                                    self.optimizer.param_groups[0]['lr'],
                                    self.current_epoch, time.time())

        self.summ_writer.add_scalar(f'{key}/sq',
                                    epoch_sq.val,
                                    self.current_epoch,
                                    time.time())

    def validate(self, mode='val'):
        """
        Model validation
        """
        if mode == 'val':
            if not self.config.data.drop_last:
                iteration_per_epoch = int(np.ceil(len(self.valid_set) /
                                                  self.valid_loader.batch_size))
            else:
                iteration_per_epoch = int(len(self.valid_set) /
                                          self.valid_loader.batch_size)
        elif mode == 'test':
            if not self.config.data.drop_last:
                iteration_per_epoch = int(np.ceil(len(self.test_set) /
                                                  self.test_loader.batch_size))
            else:
                iteration_per_epoch = int(len(self.test_set) /
                                          self.test_loader.batch_size)
        else:
            self.logger.error('validate mode is not implement, only has val mode and test mode')
            raise RuntimeError

        with torch.no_grad():                        
            if mode == 'val':
                tqdm_batch = tqdm(self.valid_loader,
                                  total=iteration_per_epoch,
                                  desc='Valid epoch {}'.format(
                                      self.current_epoch))
            else:
                tqdm_batch = tqdm(self.test_loader,
                                  total=iteration_per_epoch,
                                  desc='Test epoch {}'.format(
                                      self.current_epoch))

            # set model into evaluation mode
            self.offset_model.eval()
            self.dist_model.eval()

            # initialize average meters
            epoch_loss = AverageMeter()
            epoch_abs = AverageMeter()
            epoch_sq = AverageMeter()

            for inputs, gt_distance, *_ in tqdm_batch:
                # inputs shape: [B, 2, 4]
                inputs = inputs.to(self.device)
                gt_distance = gt_distance.to(self.device)
                # gt_distance = gt_distance / self.config.data.max_distance

                distance = self.base_forward(inputs)

                # loss function
                curr_loss = self.loss(distance, gt_distance)

                # current metrics
                # batched_abs_rel and curr_sq_rel is calculate on average of batch.
                # batched_abs_rel : [B, ]
                batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
                batched_sq_rel = compute_square_rel_error(distance, gt_distance)

                if torch.isnan(curr_loss):
                    self.logger.error('Loss is NaN during validation...')
                    raise RuntimeError

                # update average meter
                epoch_loss.update(curr_loss.item())
                for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                    epoch_abs.update(abs_rel.item())
                    epoch_sq.update(sq_rel.item())

                batched_abs_std = batched_abs_rel.std().item()
                if mode == 'val':
                    self.summ_writer.add_scalar('val/abs_std',
                                                batched_abs_std,
                                                self.val_iter, time.time())
                    self.val_iter += 1
                else:
                    self.summ_writer.add_scalar('test/abs_std',
                                                batched_abs_std,
                                                self.test_iter, time.time())
                    self.test_iter += 1

            tqdm_batch.close()

            if mode == 'val':
                self._logger(key='val',
                             epoch_dist_loss=epoch_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)

                self.logger.info('NUM: %d | Valid epoch: %d | loss: %f | ABS: %f | SQ: %f',
                                 len(self.valid_set),
                                 self.current_epoch,
                                 epoch_loss.val,
                                 epoch_abs.val,
                                 epoch_sq.val)
            else:
                self._logger(key='test',
                             epoch_dist_loss=epoch_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)
                self.logger.info('NUM: %d | Test epoch: %d | loss: %f | ABS: %f | SQ: %f',
                                 len(self.test_set),
                                 self.current_epoch,
                                 epoch_loss.val,
                                 epoch_abs.val,
                                 epoch_sq.val)

            return epoch_loss.val, epoch_abs.val

    def inference_res(self, mode='val'):
        if mode == 'val':
            if not self.config.data.drop_last:
                iteration_per_epoch = int(np.ceil(len(self.valid_set) /
                                                  self.valid_loader.batch_size))
            else:
                iteration_per_epoch = int(len(self.valid_set) /
                                          self.valid_loader.batch_size)
        elif mode == 'test':
            if not self.config.data.drop_last:
                iteration_per_epoch = int(np.ceil(len(self.test_set) /
                                                  self.test_loader.batch_size))
            else:
                iteration_per_epoch = int(len(self.test_set) /
                                          self.test_loader.batch_size)
        else:
            self.logger.error('validate mode is not implement, only has val mode and test mode')
            raise RuntimeError

        with torch.no_grad():
            if mode == 'val':
                tqdm_batch = tqdm(self.valid_loader,
                                  total=iteration_per_epoch,
                                  desc='Valid epoch {}'.format(
                                      self.current_epoch))
            else:
                tqdm_batch = tqdm(self.test_loader,
                                  total=iteration_per_epoch,
                                  desc='Test epoch {}'.format(
                                      self.current_epoch))

            # set model into evaluation mode
            self.offset_model.eval()
            self.dist_model.eval()

            # initialize average meters
            epoch_loss = AverageMeter()
            epoch_abs = AverageMeter()
            epoch_sq = AverageMeter()

            self.save_dict = []
            for inputs, gt_distance, file_id in tqdm_batch:
                # inputs shape: [B, 2, 4]
                inputs = inputs.to(self.device)
                gt_distance = gt_distance.to(self.device)
                # gt_distance = gt_distance / self.config.data.max_distance

                distance = self.base_forward(inputs)

                # loss function
                curr_loss = self.loss(distance, gt_distance)

                # current metrics
                # batched_abs_rel and curr_sq_rel is calculate on average of batch.
                # batched_abs_rel : [B, ]
                batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
                batched_sq_rel = compute_square_rel_error(distance, gt_distance)

                if torch.isnan(curr_loss):
                    self.logger.error('Loss is NaN during validation...')
                    raise RuntimeError

                # update average meter
                epoch_loss.update(curr_loss.item())
                for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                    epoch_abs.update(abs_rel.item())
                    epoch_sq.update(sq_rel.item())

                self.create_save_row(files=file_id,
                                     gt_distance=gt_distance,
                                     pred_distance=distance)

            tqdm_batch.close()
            metric_dict = {"abs_rel": [epoch_abs.val], "sq_rel": [epoch_sq.val]}
            return self.save_dict, metric_dict

    def create_save_row(self, files, gt_distance, pred_distance):
        for file_i, gt_i, pred_i in zip(files,
                                                           gt_distance,
                                                           pred_distance):
            save_row = {"file_id": file_i,
                        "gt_distance": gt_i.item(),
                        "pred_distance": pred_i.item()}
            self.save_dict.append(save_row)

    def finalize(self):
        self.logger.info('Running finalize operation...')
        self.summ_writer.close()

        self.resume(ckpt_name='best_ckpt.pth')

        val_save_list, val_save_metric = self.inference_res(mode='val')
        test_save_list, test_save_metric = self.inference_res(mode='test')

        val_res = pd.DataFrame(val_save_list)
        val_metric = pd.DataFrame(val_save_metric)

        test_res = pd.DataFrame(test_save_list)
        test_metric = pd.DataFrame(test_save_metric)

        val_res.to_csv(join(self.config.env.out_dir, 'final_val_res.csv'))
        val_metric.to_csv(join(self.config.env.out_dir, 'final_val_metric.csv'))
        test_res.to_csv(join(self.config.env.out_dir, 'final_test_res.csv'))
        test_metric.to_csv(join(self.config.env.out_dir, 'final_test_metric.csv'))
        print("Done")
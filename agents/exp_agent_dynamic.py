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

        self.val_iter = 0
        self.test_iter = 0

        self.stg2_train_iter = 0
        self.stg2_val_iter = 0
        self.stg2_test_iter = 0

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
            #　print_cuda_statistics()
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

        self.gate_model = get_model(self.config.model.gate)
        self.gate_model = self.gate_model.to(self.device)

        self.offset_model_2 = get_model(self.config.model.offset)
        self.offset_model_2 = self.offset_model_2.to(self.device)

        self.dist_model = get_model(self.config.model.dist)
        self.dist_model = self.dist_model.to(self.device)

        self.loss = get_loss_module(self.config)
        self.logger.info('processing: get loss module')
        self.loss = self.loss.to(self.device)

        self.gate_loss = get_loss_module(self.config.model.gate)
        self.logger.info('processing: get classfier loss module')
        self.gate_loss = self.gate_loss.to(self.device)

        self.optimizer = get_optimizer(self.config,
                                       self.offset_model.parameters())

        self.optimizer_g = get_optimizer(self.config,
                                       self.gate_model.parameters())

        self.optimizer_2 = get_optimizer(self.config,
                                       self.offset_model_2.parameters())


        self.scheduler = get_lr_schedule(self.config, self.optimizer)
        self.scheduler_2 = get_lr_schedule(self.config, self.optimizer_2)

        # try to load existing ckpt to resume a interrupted training
        self.resume_all_model(self.config.ckpt.ckpt_name)

    def resume(self, model_name='offset', ckpt_suffix='latest_ckpt.pth'):
        # offset_latest_ckpt.pth
        ckpt_name = model_name + '_' + ckpt_suffix
        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        try:
            self.load_ckpt(ckpt_path, model_name)
        except:
            self.logger.info('Can not load ckpt at "%s"', ckpt_path)

    def load_ckpt(self, ckpt_path, model_name='offset', strict=False):
        """
        Load checkpoint with given ckpt_name

        Args:
            ckpt_path (string): the path to ckpt
            strict (bool): whether or not to strictly load ckpt
        """

        try:
            if model_name == 'offset' or model_name == 'best_offset':
                self.logger.info('Loading ckpt from %s', ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=self.device)

                self.current_epoch = ckpt['current_epoch']
                self.current_iteration = ckpt['current_iteration']

                self.offset_model.load_state_dict(ckpt['model_state_dict'], strict=strict)
                # NOTE
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

                self.logger.info('Successfully loaded ckpt from %s at '
                                 'epoch %d, iteration %d', ckpt_path,
                                 self.current_epoch, self.current_iteration)
                self.logger.info(
                    'Loaded initial learning rate %f from ckpt',
                    ckpt['optimizer_state_dict']['param_groups'][0]['lr']
                )
            elif model_name == 'offset_2' or model_name == 'best_offset_2':
                self.logger.info('Loading ckpt from %s', ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=self.device)

                self.stg2_train_iter = ckpt['current_iteration']

                self.offset_model_2.load_state_dict(ckpt['model_state_dict'], strict=strict)
                # NOTE
                self.optimizer_2.load_state_dict(ckpt['optimizer_state_dict'])

                self.logger.info('Successfully loaded ckpt from %s at '
                                 'epoch %d, iteration %d', ckpt_path,
                                 self.current_epoch, self.stg2_train_iter)
                self.logger.info(
                    'Loaded initial learning rate %f from ckpt',
                    ckpt['optimizer_state_dict']['param_groups'][0]['lr']
                )
            elif model_name == 'gate' or model_name == 'best_gate':
                self.logger.info('Loading ckpt from %s', ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=self.device)

                self.gate_model.load_state_dict(ckpt['model_state_dict'], strict=strict)
                # NOTE
                self.optimizer_g.load_state_dict(ckpt['optimizer_state_dict'])

                self.logger.info('Successfully loaded ckpt from %s at '
                                 'epoch %d, iteration %d', ckpt_path,
                                 self.current_epoch, self.stg2_train_iter)
                self.logger.info(
                    'Loaded initial learning rate %f from ckpt',
                    ckpt['optimizer_state_dict']['param_groups'][0]['lr']
                )
        except OSError:
            self.logger.warning('No ckpt exists at "%s". Skipping...', ckpt_path)

    def save_ckpt(self, model_name='offset', ckpt_suffix='ckpt.pth', is_best=False):
        """
        Save the current state_dict of agent model to ckpt_path

        Args:
            ckpt_name (string, optional): the name of the current state_dict to save
            is_best (bool, optional): indicator for whether the model is best
        """
        ckpt_name = model_name + '_' + ckpt_suffix
        if model_name == 'offset':
            state = {'current_epoch': self.current_epoch,
                     'current_iteration': self.current_iteration,
                     'model_state_dict': self.offset_model.state_dict(),
                     'optimizer_state_dict': self.optimizer.state_dict()}
        elif model_name == 'offset_2':
            state = {'current_epoch': self.current_epoch,
                     'current_iteration': self.stg2_train_iter,
                     'model_state_dict': self.offset_model_2.state_dict(),
                     'optimizer_state_dict': self.optimizer_2.state_dict()}
        elif model_name == 'gate':
            state = {'current_epoch': self.current_epoch,
                     'model_state_dict': self.gate_model.state_dict(),
                     'optimizer_state_dict': self.optimizer_g.state_dict()}

        ckpt_path = join(self.config.env.ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)
        if is_best:
            best_ckpt_path = join(self.config.env.ckpt_dir,
                                  'best_' + ckpt_name)
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def save_all_model(self, ckpt_suffix='ckpt.pth', is_best=False):
        self.save_ckpt(model_name='offset', ckpt_suffix=ckpt_suffix, is_best=is_best)
        self.save_ckpt(model_name='offset_2', ckpt_suffix=ckpt_suffix, is_best=is_best)
        self.save_ckpt(model_name='gate', ckpt_suffix=ckpt_suffix, is_best=is_best)

    def resume_all_model(self, ckpt_suffix='latest_ckpt.pth', is_best=False):
        if is_best:
            self.resume(model_name='best_offset', ckpt_suffix=ckpt_suffix)
            self.resume(model_name='best_offset_2', ckpt_suffix=ckpt_suffix)
            self.resume(model_name='best_gate', ckpt_suffix=ckpt_suffix)
        else:
            self.resume(model_name='offset', ckpt_suffix=ckpt_suffix)
            self.resume(model_name='offset_2', ckpt_suffix=ckpt_suffix)
            self.resume(model_name='gate', ckpt_suffix=ckpt_suffix)

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
        # self.resume('debug.pth')
        # self.resume_2('debug.pth')
        for i in range(self.current_epoch, self.config.optimizer.max_epoch):
            self.train_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
                self.scheduler_2.step()
            else:
                self.logger.info('lr scheduler is None')

            if i % self.config.ckpt.save_interval == 0:
                abs_rel = self.validate()
                is_best = abs_rel < self.best_valid_metric

                if is_best:
                    self.best_valid_metric = abs_rel
                    self.save_all_model(is_best=is_best)
                    """
                    test only model has a lowest abs rel on val set
                    """
                self.validate(mode='test')

                ckpt_suffix = 'ckpt_' + str(self.current_epoch) + '.pth'
                self.save_all_model(ckpt_suffix=ckpt_suffix)
            self.current_epoch += 1

    def base_forward(self, inputs_xywh, stage=1):
        B = inputs_xywh.size()[0]
        copy_inputs = inputs_xywh.clone().detach()

        # inputs_xx shape: [B, 2, 1]
        # every batch contains one left x and one right x
        inputs_xx = inputs_xywh[:, :, 0:1]
        inputs_xx = torch.reshape(inputs_xx, (B, 2))

        # inputs_adwh: inputs containing: angle, distance(to image center), width, height
        inputs_adwh = self.train_set.cal_rotate_dist(inputs_xywh, is_norm=self.config.data.norm_rad)  # re [2*B, 4]

        inputs_adwh = inputs_adwh.reshape(2 * B, 1, -1, 1)  # [2*B, 1, 4, 1]
        """
        # inputs_wh shape: [B, 2, 2]
        inputs_wh = inputs[:, :, :]

        # inputs_wh shape: [2*B, 2, 1]
        # conv1d [Batch, Channel, length]
        inputs_wh = torch.reshape(inputs_wh, (2 * inputs.shape[0], 1, -1, 1))
        """
        if stage == 1:
            # forward propagation
            # offset_preds shape: [2*B, 4, 1]
            offset_preds = self.offset_model(inputs_adwh)  # return [2*B, 1]

            # offset_preds shape: [B, 2]
            offset_preds = torch.reshape(offset_preds, (B, -1))  # return [B, 2]
            distance, left_modified_x, right_modified_x = self.dist_model(offset_preds, inputs_xx)
            # modified inputs
            copy_inputs[:, 0, 0] = left_modified_x.detach()
            copy_inputs[:, 1, 0] = right_modified_x.detach()

            return copy_inputs, distance
        elif stage == 2:
            # forward propagation
            # offset_preds shape: [2*B, 4, 1]
            offset_preds = self.offset_model_2(inputs_adwh)  # return [2*B, 1]
            # offset_preds shape: [B, 2]
            offset_preds = torch.reshape(offset_preds, (B, -1))  # return [B, 2]
            distance, left_modified_x, right_modified_x = self.dist_model(offset_preds, inputs_xx)

            copy_inputs[:, 0, 0] = left_modified_x.detach()
            copy_inputs[:, 1, 0] = right_modified_x.detach()

            return copy_inputs, distance
        elif stage == 'gate':
            gate_preds = self.gate_model(inputs_adwh)  # return [2*B, 2]
            gate_preds = gate_preds.reshape((B, 2, -1))
            # gate_pred = torch.mean(cls_pred, dim=1)
            if self.config.gate_head == 'mean':
                gate_preds = (gate_preds[:, 0, :] + gate_preds[:, 1, :]) / 2
            elif self.config.gate_head == 'max':
                gate_preds = torch.max(gate_preds[:, 0, :], gate_preds[:, 1, :])
            else:
                raise NotImplementedError(f'More Stage is not implement')

            return gate_preds
        else:
            raise NotImplementedError(f'More Stage is not implement')

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
        self.gate_model.train()
        self.offset_model_2.train()

        # initialize average meters
        epoch_dist_loss = AverageMeter()
        epoch_gate_loss = AverageMeter()
        epoch_abs = AverageMeter()
        epoch_sq = AverageMeter()

        stg2_epoch_dist_loss = AverageMeter()
        stg2_epoch_abs = AverageMeter()
        stg2_epoch_sq = AverageMeter()

        stg2_num = 0
        gate_tp = 0
        # stare the training loop over iterations of one epoch
        for inputs, gt_distance, file_id in tqdm_batch:
            # inputs shape: [B, 2, 4]
            inputs = inputs.to(self.device)
            gt_distance = gt_distance.to(self.device)

            modified_inputs, distance = self.base_forward(inputs, stage=1)
            # current metrics
            # batched_abs_rel and curr_sq_rel is calculated on average of batch.
            # batched_abs_rel : [B, ]
            batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
            batched_sq_rel = compute_square_rel_error(distance, gt_distance)
            dist_loss = self.loss(distance, gt_distance)
            # optimizer
            self.optimizer.zero_grad()
            dist_loss.backward()

            if self.config.optimizer.clip_grad.is_clip:
                torch.nn.utils.clip_grad_norm_(self.offset_model.parameters(),
                                               self.config.optimizer.clip_grad.max_grad_norm)
            self.optimizer.step()

            if torch.isnan(dist_loss):
                self.logger.error('Loss is NaN during training...')
                raise RuntimeError

            if self.current_epoch == self.config.start_dynamic and self.config.best_init:
                self.resume_2('best_ckpt.pth')
            if self.current_epoch > self.config.start_dynamic:
                hard_sample_mask = batched_abs_rel > self.config.threshold_abs
                # Stage 2 training
                hard_sample = modified_inputs[hard_sample_mask]
                if hard_sample.size()[0] > 0:
                    stg2_num += hard_sample.size()[0]

                    # training stage 1 gate when hard sample exists.
                    gate_preds = self.base_forward(inputs, stage='gate')

                    gate_label = torch.zeros_like(gt_distance)
                    # label hard sample to be 1
                    gate_label[hard_sample_mask] = 1
                    gate_loss = self.gate_loss(gate_preds, gate_label.long())

                    self.optimizer_g.zero_grad()
                    gate_loss.backward()
                    if self.config.optimizer.clip_grad.is_clip:
                        torch.nn.utils.clip_grad_norm_(self.gate_model.parameters(),
                                                       self.config.optimizer.clip_grad.max_grad_norm)
                    self.optimizer_g.step()
                    epoch_gate_loss.update(gate_loss.item())

                    # begining stage 2 training
                    hard_sample_gt = gt_distance[hard_sample_mask]
                    stg2_modified_inputs, stg2_distance = self.base_forward(hard_sample, stage=2)
                    stg2_dist_loss = self.loss(stg2_distance, hard_sample_gt)
                    stg2_batched_abs_rel = compute_abs_rel_error(stg2_distance, hard_sample_gt)
                    stg2_batched_sq_rel = compute_square_rel_error(stg2_distance, hard_sample_gt)

                    self.optimizer_2.zero_grad()
                    stg2_dist_loss.backward()
                    if self.config.optimizer.clip_grad.is_clip:
                        torch.nn.utils.clip_grad_norm_(self.offset_model_2.parameters(),
                                                       self.config.optimizer.clip_grad.max_grad_norm)
                    self.optimizer_2.step()
                    # update average meter
                    stg2_epoch_dist_loss.update(stg2_dist_loss.item())
                    for abs_rel, sq_rel in zip(stg2_batched_abs_rel, stg2_batched_sq_rel):
                        stg2_epoch_abs.update(abs_rel.item())
                        stg2_epoch_sq.update(sq_rel.item())

                    if stg2_batched_abs_rel.size()[0] > 1:
                        stg2_batched_abs_std = stg2_batched_abs_rel.std()
                        self.summ_writer.add_scalar('train_2/abs_std',
                                                    stg2_batched_abs_std,
                                                    self.stg2_train_iter, time.time())
                    self.stg2_train_iter += 1

            # update average meter
            epoch_dist_loss.update(dist_loss.item())
            for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                epoch_abs.update(abs_rel.item())
                epoch_sq.update(sq_rel.item())

            batched_abs_std = batched_abs_rel.std()

            self.summ_writer.add_scalar('train_1/abs_std',
                                        batched_abs_std,
                                        self.current_iteration, time.time())
            self.current_iteration += 1

        tqdm_batch.close()
        self._logger(key='train_1',
                     epoch_dist_loss=epoch_dist_loss,
                     epoch_cls_loss=epoch_gate_loss,
                     epoch_abs=epoch_abs,
                    epoch_sq=epoch_sq)
        self.logger.info('NUM: %d | train_1 epoch: %d | lr: %f | dist loss: %f | gate loss: %f | ABS: %f | SQ: %f',
                         len(self.train_set),
                         self.current_epoch,
                         self.optimizer.param_groups[0]['lr'],
                         epoch_dist_loss.val,
                         epoch_gate_loss.val,
                         epoch_abs.val,
                         epoch_sq.val)

        if self.current_epoch > self.config.start_dynamic:
            self._logger(key='train_2',
                         epoch_dist_loss=stg2_epoch_dist_loss,
                         epoch_abs=stg2_epoch_abs,
                         epoch_sq=stg2_epoch_sq,
                         stage=2)
            self.summ_writer.add_scalar('train_2/stg2_num',
                                        stg2_num,
                                        self.current_epoch, time.time())

            self.logger.info('NUM: %d | train_2 epoch: %d | lr: %f | dist loss: %f | ABS: %f | SQ: %f',
                             stg2_num,
                             self.current_epoch,
                             self.optimizer_2.param_groups[0]['lr'],
                             stg2_epoch_dist_loss.val,
                             stg2_epoch_abs.val,
                             stg2_epoch_sq.val)

    def _logger(self,
                key,
                epoch_dist_loss,
                epoch_abs,
                epoch_sq,
                epoch_cls_loss=None,
                stage=1):
        self.summ_writer.add_scalar(f'{key}/dist_loss',
                                    epoch_dist_loss.val,
                                    self.current_epoch, time.time())
        if stage == 1:
            self.summ_writer.add_scalar(f'{key}/gate_loss',
                                        epoch_cls_loss.val,
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
            self.offset_model_2.eval()
            self.gate_model.eval()
            self.dist_model.eval()

            # initialize average meters
            # stg_1_meters
            epoch_dist_loss = AverageMeter()
            epoch_gate_loss = AverageMeter()
            epoch_abs = AverageMeter()
            epoch_sq = AverageMeter()
            # stg_2_meters
            stg2_epoch_dist_loss = AverageMeter()
            stg2_epoch_abs = AverageMeter()
            stg2_epoch_sq = AverageMeter()
            stg2_num = 0

            # final_meters
            final_epoch_abs = AverageMeter()
            final_epoch_sq = AverageMeter()
            for inputs, gt_distance, *_ in tqdm_batch:
                # inputs shape: [B, 2, 4]
                inputs = inputs.to(self.device)
                gt_distance = gt_distance.to(self.device)
                # gt_distance = gt_distance / self.config.data.max_distance

                modified_inputs, distance = self.base_forward(inputs, stage=1)
                # current metrics
                # batched_abs_rel and curr_sq_rel is calculate on average of batch.
                # batched_abs_rel : [B, ]
                batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
                batched_sq_rel = compute_square_rel_error(distance, gt_distance)
                dist_loss = self.loss(distance, gt_distance)
                if torch.isnan(dist_loss):
                    self.logger.error('Loss is NaN during validation...')
                    raise RuntimeError

                # update average meter
                epoch_dist_loss.update(dist_loss.item())
                for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                    epoch_abs.update(abs_rel.item())
                    epoch_sq.update(sq_rel.item())

                if self.current_epoch > self.config.start_dynamic:
                    gate_preds = self.base_forward(inputs, stage='gate')
                    # This part is to calculate loss
                    hard_sample_mask = batched_abs_rel > self.config.threshold_abs
                    gate_label = torch.zeros_like(gt_distance)
                    # label hard sample to be 1
                    gate_label[hard_sample_mask] = 1
                    gate_loss = self.gate_loss(gate_preds, gate_label.long())
                    epoch_gate_loss.update(gate_loss.item())

                    # This is for choosing hard sample for stage 2 model.
                    stg2_sample_mask = gate_preds.argmax(dim=1).bool()
                    hard_sample = modified_inputs[stg2_sample_mask]
                    if hard_sample.size()[0] > 0:
                        stg2_num += hard_sample.size()[0]
                        # begining stage 2 training
                        hard_sample_gt = gt_distance[stg2_sample_mask]

                        stg2_modified_inputs, stg2_distance = self.base_forward(hard_sample, stage=2)

                        stg2_dist_loss = self.loss(stg2_distance, hard_sample_gt)
                        stg2_batched_abs_rel = compute_abs_rel_error(stg2_distance, hard_sample_gt)
                        stg2_batched_sq_rel = compute_square_rel_error(stg2_distance, hard_sample_gt)

                        # update average meter
                        stg2_epoch_dist_loss.update(stg2_dist_loss.item())
                        for abs_rel, sq_rel in zip(stg2_batched_abs_rel, stg2_batched_sq_rel):
                            stg2_epoch_abs.update(abs_rel.item())
                            stg2_epoch_sq.update(sq_rel.item())
                            # final_res
                            final_epoch_abs.update(abs_rel.item())
                            final_epoch_sq.update(sq_rel.item())

                        if stg2_batched_abs_rel.size()[0] > 1:
                            stg2_batched_abs_std = stg2_batched_abs_rel.std()
                            if mode == 'val':
                                self.summ_writer.add_scalar('val_2/abs_std',
                                                            stg2_batched_abs_std,
                                                            self.stg2_val_iter, time.time())
                            elif mode == 'test':
                                self.summ_writer.add_scalar('test_2/abs_std',
                                                            stg2_batched_abs_std,
                                                            self.stg2_test_iter, time.time())
                        if mode == 'val':
                            self.stg2_val_iter += 1
                        elif mode == 'test':
                            self.stg2_test_iter += 1

                if self.current_epoch > self.config.start_dynamic:
                    for abs_rel, sq_rel in zip(batched_abs_rel[~stg2_sample_mask], batched_sq_rel[~stg2_sample_mask]):
                        final_epoch_abs.update(abs_rel.item())
                        final_epoch_sq.update(sq_rel.item())
                else:
                    for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                        final_epoch_abs.update(abs_rel.item())
                        final_epoch_sq.update(sq_rel.item())

                batched_abs_std = batched_abs_rel.std()
                if mode == 'val':
                    self.summ_writer.add_scalar('val_1/abs_std',
                                                batched_abs_std,
                                                self.current_iteration, time.time())
                    self.val_iter += 1
                elif mode == 'test':
                    self.summ_writer.add_scalar('test_1/abs_std',
                                                batched_abs_std,
                                                self.current_iteration, time.time())
                    self.test_iter += 1
            tqdm_batch.close()

            if mode == 'val':
                self._logger(key='val_1',
                             epoch_dist_loss=epoch_dist_loss,
                             epoch_cls_loss=epoch_gate_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)
                self.summ_writer.add_scalar('val/final_abs',
                                            final_epoch_abs.val,
                                            self.current_epoch, time.time())
                self.summ_writer.add_scalar('val/final_sq',
                                            final_epoch_sq.val,
                                            self.current_epoch, time.time())
                self.logger.info(
                    'NUM: %d | val_1 epoch: %d | lr: %f | dist loss: %f | gate loss: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    epoch_dist_loss.val,
                    epoch_gate_loss.val,
                    epoch_abs.val,
                    epoch_sq.val)

                self.logger.info(
                    'NUM: %d | Val Final epoch: %d | lr: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    final_epoch_abs.val,
                    final_epoch_sq.val)

            else:
                self._logger(key='test_1',
                             epoch_dist_loss=epoch_dist_loss,
                             epoch_cls_loss=epoch_gate_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)
                self.summ_writer.add_scalar('test/final_abs',
                                            final_epoch_abs.val,
                                            self.current_epoch, time.time())
                self.summ_writer.add_scalar('test/final_sq',
                                            final_epoch_sq.val,
                                            self.current_epoch, time.time())
                self.logger.info(
                    'NUM: %d | test_1 epoch: %d | lr: %f | dist loss: %f | gate loss: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    epoch_dist_loss.val,
                    epoch_gate_loss.val,
                    epoch_abs.val,
                    epoch_sq.val)

                self.logger.info(
                    'NUM: %d | Test Final epoch: %d | lr: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    final_epoch_abs.val,
                    final_epoch_sq.val)

            if self.current_epoch > self.config.start_dynamic:
                if mode == 'val':
                    self._logger(key='val_2',
                                 epoch_dist_loss=stg2_epoch_dist_loss,
                                 epoch_abs=stg2_epoch_abs,
                                 epoch_sq=stg2_epoch_sq,
                                 stage=2)
                    self.summ_writer.add_scalar('val_2/stg2_num',
                                                stg2_num,
                                                self.current_epoch, time.time())

                    self.logger.info(
                        'NUM: %d | val_2 epoch: %d | lr: %f | dist loss: %f | ABS: %f | SQ: %f',
                        stg2_num,
                        self.current_epoch,
                        self.optimizer_2.param_groups[0]['lr'],
                        stg2_epoch_dist_loss.val,
                        stg2_epoch_abs.val,
                        stg2_epoch_sq.val)
                else:
                    self._logger(key='test_2',
                                 epoch_dist_loss=stg2_epoch_dist_loss,
                                 epoch_abs=stg2_epoch_abs,
                                 epoch_sq=stg2_epoch_sq,
                                 stage=2)
                    self.summ_writer.add_scalar('test_2/stg2_num',
                                                stg2_num,
                                                self.current_epoch, time.time())

                    self.logger.info(
                        'NUM: %d | test_2 epoch: %d | lr: %f | dist loss: %f | ABS: %f | SQ: %f',
                        stg2_num,
                        self.current_epoch,
                        self.optimizer_2.param_groups[0]['lr'],
                        stg2_epoch_dist_loss.val,
                        stg2_epoch_abs.val,
                        stg2_epoch_sq.val)
            return final_epoch_abs.val

    def inference(self, mode='val'):
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
            self.offset_model_2.eval()
            self.gate_model.eval()
            self.dist_model.eval()

            # initialize average meters
            # stg_1_meters
            epoch_dist_loss = AverageMeter()
            epoch_gate_loss = AverageMeter()
            epoch_abs = AverageMeter()
            epoch_sq = AverageMeter()
            # stg_2_meters
            stg2_epoch_dist_loss = AverageMeter()
            stg2_epoch_abs = AverageMeter()
            stg2_epoch_sq = AverageMeter()
            stg2_num = 0

            # final_meters
            final_epoch_abs = AverageMeter()
            final_epoch_sq = AverageMeter()

            self.save_dict = []

            for inputs, gt_distance, file_id in tqdm_batch:
                # inputs shape: [B, 2, 4]
                inputs = inputs.to(self.device)
                gt_distance = gt_distance.to(self.device)
                # gt_distance = gt_distance / self.config.data.max_distance

                modified_inputs, distance = self.base_forward(inputs, stage=1)
                # current metrics
                # batched_abs_rel and curr_sq_rel is calculate on average of batch.
                # batched_abs_rel : [B, ]
                batched_abs_rel = compute_abs_rel_error(distance, gt_distance)
                batched_sq_rel = compute_square_rel_error(distance, gt_distance)
                dist_loss = self.loss(distance, gt_distance)
                if torch.isnan(dist_loss):
                    self.logger.error('Loss is NaN during validation...')
                    raise RuntimeError

                # update average meter
                epoch_dist_loss.update(dist_loss.item())
                for abs_rel, sq_rel in zip(batched_abs_rel, batched_sq_rel):
                    epoch_abs.update(abs_rel.item())
                    epoch_sq.update(sq_rel.item())

                gate_preds = self.base_forward(inputs, stage='gate')
                # This part is to calculate loss
                hard_sample_mask = batched_abs_rel > self.config.threshold_abs
                gate_label = torch.zeros_like(gt_distance)
                # label hard sample to be 1
                gate_label[hard_sample_mask] = 1
                gate_loss = self.gate_loss(gate_preds, gate_label.long())
                epoch_gate_loss.update(gate_loss.item())

                # This is for choosing hard sample for stage 2 model.
                stg2_sample_mask = gate_preds.argmax(dim=1).bool()
                hard_sample = modified_inputs[stg2_sample_mask]
                if hard_sample.size()[0] > 0:
                    stg2_num += hard_sample.size()[0]
                    # begining stage 2 training
                    hard_sample_gt = gt_distance[stg2_sample_mask]

                    stg2_modified_inputs, stg2_distance = self.base_forward(hard_sample, stage=2)

                    stg2_dist_loss = self.loss(stg2_distance, hard_sample_gt)
                    stg2_batched_abs_rel = compute_abs_rel_error(stg2_distance, hard_sample_gt)
                    stg2_batched_sq_rel = compute_square_rel_error(stg2_distance, hard_sample_gt)

                    # update average meter
                    stg2_epoch_dist_loss.update(stg2_dist_loss.item())
                    for abs_rel, sq_rel in zip(stg2_batched_abs_rel, stg2_batched_sq_rel):
                        stg2_epoch_abs.update(abs_rel.item())
                        stg2_epoch_sq.update(sq_rel.item())
                        # final_res
                        final_epoch_abs.update(abs_rel.item())
                        final_epoch_sq.update(sq_rel.item())

                    self.create_save_row(list(compress(file_id, stg2_sample_mask)),
                                         inputs[stg2_sample_mask],
                                         stg2_modified_inputs,
                                         gt_distance[stg2_sample_mask],
                                         stg2_distance)

                for abs_rel, sq_rel in zip(batched_abs_rel[~stg2_sample_mask], batched_sq_rel[~stg2_sample_mask]):
                    final_epoch_abs.update(abs_rel.item())
                    final_epoch_sq.update(sq_rel.item())

                self.create_save_row(list(compress(file_id, ~stg2_sample_mask)),
                                     inputs[~stg2_sample_mask],
                                     modified_inputs[~stg2_sample_mask],
                                     gt_distance[~stg2_sample_mask],
                                     distance[~stg2_sample_mask])
            tqdm_batch.close()

            if mode == 'val':
                self._logger(key='val_1',
                             epoch_dist_loss=epoch_dist_loss,
                             epoch_cls_loss=epoch_gate_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)
                metric_dict = {"abs_rel": [final_epoch_abs.val], "sq_rel": [final_epoch_sq.val]}
                self.logger.info(
                    'NUM: %d | val_1 epoch: %d | lr: %f | dist loss: %f | gate loss: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    epoch_dist_loss.val,
                    epoch_gate_loss.val,
                    epoch_abs.val,
                    epoch_sq.val)
                self.logger.info(
                    'NUM: %d | Val Final epoch: %d | lr: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    final_epoch_abs.val,
                    final_epoch_sq.val)

            else:
                self._logger(key='test_1',
                             epoch_dist_loss=epoch_dist_loss,
                             epoch_cls_loss=epoch_gate_loss,
                             epoch_abs=epoch_abs,
                             epoch_sq=epoch_sq)
                metric_dict = {"abs_rel": [final_epoch_abs.val], "sq_rel": [final_epoch_sq.val]}
                self.logger.info(
                    'NUM: %d | test_1 epoch: %d | lr: %f | dist loss: %f | gate loss: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    epoch_dist_loss.val,
                    epoch_gate_loss.val,
                    epoch_abs.val,
                    epoch_sq.val)

                self.logger.info(
                    'NUM: %d | Test Final epoch: %d | lr: %f | ABS: %f | SQ: %f',
                    len(self.valid_set),
                    self.current_epoch,
                    self.optimizer.param_groups[0]['lr'],
                    final_epoch_abs.val,
                    final_epoch_sq.val)

            if mode == 'val':
                self._logger(key='val_2',
                             epoch_dist_loss=stg2_epoch_dist_loss,
                             epoch_abs=stg2_epoch_abs,
                             epoch_sq=stg2_epoch_sq,
                             stage=2)
                metric_dict["stag2_num"] = [stg2_num]

                self.logger.info(
                    'NUM: %d | val_2 epoch: %d | lr: %f | dist loss: %f | ABS: %f | SQ: %f',
                    stg2_num,
                    self.current_epoch,
                    self.optimizer_2.param_groups[0]['lr'],
                    stg2_epoch_dist_loss.val,
                    stg2_epoch_abs.val,
                    stg2_epoch_sq.val)
            else:
                self._logger(key='test_2',
                             epoch_dist_loss=stg2_epoch_dist_loss,
                             epoch_abs=stg2_epoch_abs,
                             epoch_sq=stg2_epoch_sq,
                             stage=2)
                metric_dict["stag2_num"] = [stg2_num]

                self.logger.info(
                    'NUM: %d | test_2 epoch: %d | lr: %f | dist loss: %f | ABS: %f | SQ: %f',
                    stg2_num,
                    self.current_epoch,
                    self.optimizer_2.param_groups[0]['lr'],
                    stg2_epoch_dist_loss.val,
                    stg2_epoch_abs.val,
                    stg2_epoch_sq.val)
            return self.save_dict, metric_dict

    def create_save_row(self, files, inputs, modified_inputs, gt_distance, pred_distance):
        for file_i, inp_i, modified_i, gt_i, pred_i in zip(files,
                                                           inputs,
                                                           modified_inputs,
                                                           gt_distance,
                                                           pred_distance):
            xl, yl, wl, hl = inp_i[0]
            xr, yr, wr, hr = inp_i[1]
            modified_xl = modified_i[0, 0]
            modified_xr = modified_i[1, 0]

            save_row = {"file_id": file_i,
                        "xl": xl.item(), "yl": yl.item(), "wl": wl.item(), "hl": hl.item(),
                        "xr": xr.item(), "yr": yr.item(), "wr": wr.item(), "hr": hr.item(),
                        "modifiex xl": modified_xl.item(),
                        "modified_xr": modified_xr.item(),
                        "gt_distance": gt_i.item(),
                        "pred_distance": pred_i.item()}
            self.save_dict.append(save_row)

    def finalize(self):
        self.logger.info('Running finalize operation...')
        self.summ_writer.close()

        self.resume_all_model(ckpt_suffix='ckpt.pth', is_best=True)

        val_save_list, val_save_metric = self.inference(mode='val')
        test_save_list, test_save_metric = self.inference(mode='test')
        val_res = pd.DataFrame(val_save_list)
        val_metric = pd.DataFrame(val_save_metric)

        test_res = pd.DataFrame(test_save_list)
        test_metric = pd.DataFrame(test_save_metric)

        val_res.to_csv(join(self.config.env.out_dir, 'final_val_res.csv'))
        val_metric.to_csv(join(self.config.env.out_dir, 'final_val_metric.csv'))

        test_res.to_csv(join(self.config.env.out_dir, 'final_test_res.csv'))
        test_metric.to_csv(join(self.config.env.out_dir, 'final_test_metric.csv'))
        print("Done")




import numpy as np
import matplotlib.pyplot as plt

import shutil
import argparse
import os
import json
import random
import warnings
from termcolor import colored
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter

import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

import misc
import dataset
from net import ResNet
from config import Config

def load_tf_weights_dict(net, tf_chkpt_path=None):
    """ for original resnet only"""
    chkpt = dict(np.load('/home/bryan/work/pretrained/ImageNet-ResNet50.npz'))
    
    net_dict = net.state_dict()
    mapped_weights_dict = {}
    for key in chkpt.keys():
        if 'linear' in key:
            continue
        map_key = ""
        key_parts = key.split("/")
        if 'group' in key:
            map_key = 'd%s.block.%s.%s' % \
                    (key_parts[0][-1],
                     key_parts[1][-1],
                     key_parts[2])  
        else: # first conv
            map_key = 'conv0'
        if 'shortcut' in key:
            map_key = map_key.replace('convshortcut', 'shorcut')
        if 'bn' in key:
            map_key += '.bn.'
            if 'gamma' in key:
                map_key += 'weight'
            if 'beta' in key:
                map_key += 'bias'
            if 'variance' in key:
                map_key += 'running_var'
            if 'mean' in key:
                map_key += 'running_mean'
        else:
            map_key += '.conv.'
            if 'W' in key:
                map_key += 'weight'
        if map_key not in net_dict:
            print(key, '\t F \t' % (map_key in net_dict), map_key)
        mapped_array = chkpt[key]
        if len(mapped_array.shape) > 2:
            mapped_array = np.transpose(mapped_array, (3, 2, 0, 1))
        mapped_weights_dict[map_key] = torch.from_numpy(mapped_array)
    
    # copy back remaining variable from initialized state
    for key in net_dict.keys():
        if key not in mapped_weights_dict:
            mapped_weights_dict[key] = net_dict[key]
    net.load_state_dict(mapped_weights_dict)

    return net

####
class Trainer(Config):
    ####
    def view_dataset(self, mode='train'):
        train_pairs, valid_pairs = dataset.prepare_data()
        if mode == 'train':
            train_augmentors = self.train_augmentors()
            ds = dataset.DatasetSerial(train_pairs,
                            shape_augs=iaa.Sequential(train_augmentors[0]),
                            input_augs=iaa.Sequential(train_augmentors[1]))
        else:
            infer_augmentors = self.infer_augmentors()
            ds = dataset.DatasetSerial(valid_pairs,
                            shape_augs=iaa.Sequential(infer_augmentors))
        dataset.visualize(ds, 4)
        return
    ####
    def train_step(self, net, batch, optimizer, device):
        net.train() # train mode

        imgs, true = batch # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long() # not one-hot

        # -----------------------------------------------------------
        net.zero_grad() # not rnn so not accumulate

        logit = net(imgs) # forward
        prob = F.softmax(logit, dim=-1)

        # has built-int log softmax so accept logit
        loss = F.cross_entropy(logit, true, reduction='mean')
        pred = torch.argmax(prob, dim=-1)
        acc  = torch.mean((pred == true).float()) # batch accuracy

        # gradient update
        loss.backward()
        optimizer.step()

        # -----------------------------------------------------------
        return dict(loss=loss.item(), 
                    acc=acc.item())
    ####
    def infer_step(self, net, batch, device):
        net.eval() # infer mode

        imgs, true = batch # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long() # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs)
            prob = nn.functional.softmax(logit, dim=-1)
            return dict(prob=prob.cpu().numpy(), 
                        true=true.cpu().numpy())
    ####
    def run_once(self, opt, log_dir, pretrained_path=None):
        """
        `pretrained_path` should lead to pytorch checkpoint
        """
        misc.check_manual_seed(self.seed)
        train_pairs, valid_pairs = dataset.prepare_data()

        # --------------------------- Dataloader

        train_augmentors = self.train_augmentors()
        train_dataset = dataset.DatasetSerial(train_pairs[:],
                        shape_augs=iaa.Sequential(train_augmentors[0]),
                        input_augs=iaa.Sequential(train_augmentors[1]))

        infer_augmentors = self.infer_augmentors()
        infer_dataset = dataset.DatasetSerial(valid_pairs[:],
                        shape_augs=iaa.Sequential(infer_augmentors[0]),
                        input_augs=iaa.Sequential(infer_augmentors[1]))

        train_loader = data.DataLoader(train_dataset, 
                                num_workers=self.nr_procs_train, 
                                batch_size=opt['train_batch_size'], 
                                shuffle=True, drop_last=True)

        valid_loader = data.DataLoader(infer_dataset, 
                                num_workers=self.nr_procs_valid, 
                                batch_size=opt['infer_batch_size'], 
                                shuffle=True, drop_last=False)

        # --------------------------- Training Sequence

        if self.logging:
            misc.check_log_dir(log_dir)

        device = 'cuda'

        # networks
        net = ResNet(freeze=opt['freeze'])

        if pretrained_path is not None:
            saved_state = torch.load(pretrained_path)
            if isinstance(saved_state, torch.nn.DataParallel):
                saved_state = saved_state.module.state_dict()
            net.load_state_dict(saved_state, strict=False) # set strict=True to debug

        net = torch.nn.DataParallel(net).to(device)

        optimizer, optimizer_args = opt['optimizer']
        optimizer = optimizer(net.parameters(), **optimizer_args)
        scheduler = opt['scheduler'](optimizer)

        trainer = Engine(lambda engine, batch: self.train_step(net, batch, optimizer, device))
        inferer = Engine(lambda engine, batch: self.infer_step(net, batch, device))

        train_output = ['loss', 'acc']
        infer_output = ['prob', 'true']
        ##

        if self.logging:
            checkpoint_handler = ModelCheckpoint(log_dir, self.chkpts_prefix, 
                                            save_interval=1, n_saved=30, require_empty=False)
            # adding handlers using `trainer.add_event_handler` method API
            trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                                    to_save={'net': net}) 

        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(inferer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        RunningAverage(alpha=0.95, output_transform=lambda x: x['loss']).attach(trainer, 'loss')
        RunningAverage(alpha=0.95, output_transform=lambda x: x['acc']).attach(trainer, 'acc')

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=['loss'])
        pbar.attach(inferer)

        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EXCEPTION_RAISED)
        def handle_exception(engine, e):
            if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
                engine.terminate()
                warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
                checkpoint_handler(engine, {'net_exception': net})
            else:
                raise e

        # writer for tensorboard logging
        if self.logging:
            writer = SummaryWriter(log_dir=log_dir)
            json_log_file = log_dir + '/stats.json'
            with open(json_log_file, 'w') as json_file:
                json.dump({}, json_file) # create empty file

        @trainer.on(Events.EPOCH_STARTED)
        def log_lrs(engine):
            if self.logging:
                lr = float(optimizer.param_groups[0]['lr'])
                writer.add_scalar("lr", lr, engine.state.epoch)
            # advance scheduler clock
            if scheduler is not None:
                scheduler.step()

        ####
        def update_logs(output, epoch, prefix, color):
            # print values and convert
            max_length = len(max(output.keys(), key=len))
            for metric in output:
                key = colored(prefix + '-' + metric.ljust(max_length), color)
                print('------%s : ' % key, end='')
                if metric != 'conf_mat':
                    print('%0.7f' % output[metric])
                else:
                    conf_mat = output['conf_mat'] # use pivot to turn back
                    conf_mat_df = pd.DataFrame(conf_mat)
                    conf_mat_df.index.name = 'True'
                    conf_mat_df.columns.name = 'Pred'
                    output['conf_mat'] = conf_mat_df
                    print('\n', conf_mat_df)
            if 'train' in prefix:
                lr = float(optimizer.param_groups[0]['lr'])
                key = colored(prefix + '-' + 'lr'.ljust(max_length), color)
                print('------%s : %0.7f' % (key, lr))

            if not self.logging:
                return

            # create stat dicts
            stat_dict = {}
            for metric in output:
                if metric != 'conf_mat':
                    metric_value = output[metric] 
                else:
                    conf_mat_df = output['conf_mat'] # use pivot to turn back
                    conf_mat_df = conf_mat_df.unstack().rename('value').reset_index()
                    conf_mat_df = pd.Series({'conf_mat' : conf_mat}).to_json(orient='records')
                    metric_value = conf_mat_df
                stat_dict['%s-%s' % (prefix, metric)] = metric_value

            # json stat log file, update and overwrite
            with open(json_log_file) as json_file:
                json_data = json.load(json_file)

            current_epoch = str(epoch)
            if current_epoch in json_data:
                old_stat_dict = json_data[current_epoch]
                stat_dict.update(old_stat_dict)
            current_epoch_dict = {current_epoch : stat_dict}
            json_data.update(current_epoch_dict)

            with open(json_log_file, 'w') as json_file:
                json.dump(json_data, json_file)

            # log values to tensorboard
            for metric in output:
                if metric != 'conf_mat':
                    writer.add_scalar(prefix + '-' + metric, output[metric], current_epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_running_results(engine):
            """
            running training measurement
            """
            training_ema_output = engine.state.metrics #
            update_logs(training_ema_output, engine.state.epoch, prefix='train-ema', color='green')

        ####
        def get_init_accumulator(output_names):
            return {metric : [] for metric in output_names}

        def process_accumulated_output(output):
            #
            def uneven_seq_to_np(seq, batch_size=opt['infer_batch_size']):
                item_count = batch_size * (len(seq) - 1) + len(seq[-1])
                cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
                for idx in range(0, len(seq)-1):
                    cat_array[idx   * batch_size : 
                            (idx+1) * batch_size] = seq[idx] 
                cat_array[(idx+1) * batch_size:] = seq[-1]
                return cat_array
            #
            prob = uneven_seq_to_np(output['prob'])
            true = uneven_seq_to_np(output['true'])
            # threshold then get accuracy
            pred = np.argmax(prob, axis=-1)
            acc = np.mean(pred == true)
            # confusion matrix
            conf_mat = confusion_matrix(true, pred, 
                                labels=np.arange(self.nr_classes))
            #
            proc_output = dict(acc=acc, conf_mat=conf_mat)
            return proc_output

        @trainer.on(Events.EPOCH_COMPLETED)
        def infer_valid(engine):
            """
            inference measurement
            """
            inferer.accumulator = get_init_accumulator(infer_output)
            inferer.run(valid_loader)
            output_stat = process_accumulated_output(inferer.accumulator)
            update_logs(output_stat, engine.state.epoch, prefix='valid', color='red')

        @inferer.on(Events.ITERATION_COMPLETED)    
        def accumulate_outputs(engine):
            batch_output = engine.state.output
            for key, item in batch_output.items():
                engine.accumulator[key].extend([item])
        ###
            
        # Setup is done. Now let's run the training
        trainer.run(train_loader, opt['nr_epochs'])
        return
    ####
    def run(self):
        def get_last_chkpt_path(phase1_dir):
            stat_file_path = phase1_dir + '/stats.json'
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            chkpt_list = [int(epoch) for epoch in info.keys()]
            last_chkpts_path = "%smodel_net_%d.pth" % (phase1_dir, max(chkpt_list))
            return last_chkpts_path

        phase_opts = self.training_phase

        if len(phase_opts) > 1:
            for idx, opt in enumerate(phase_opts):
                log_dir = '%s/%02d/' % (self.log_dir, idx)
                pretrained_path = opt['pretrained'] 
                if pretrained_path == -1:
                    pretrained_path = get_last_chkpt_path(prev_log_dir)
                self.run_once(opt, log_dir, pretrained_path)
                prev_log_dir = log_dir
        else:
            opt = phase_opts[0]
            self.run_once(opt, self.log_dir, opt['pretrained'])

        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    args = parser.parse_args()

    trainer = Trainer()
    if args.view:
        trainer.view_dataset()
        exit()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    trainer.run()
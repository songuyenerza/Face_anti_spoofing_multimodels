# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm
import os
import torch
torch.cuda.is_available()
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader, get_val_loader


from src.utility import get_kernel, parse_model_name
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.val_loader = get_val_loader(self.conf, root_path= conf.val_root_path)


    def train_model(self):
        self._init_model_param()
        self._train_stage()


    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()

        model = self._define_network()
        # model = self.model
        model.prob = nn.Identity() 
        model.prob = nn.Linear(in_features=128, out_features=2, bias=True)

        device = torch.device('cuda:0')

        model.to(device)
        self.model = model
        # print("model===", model)

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = 0.
        # running_loss_cls = 0.
        # running_loss_ft = 0.
        is_first = True

        loss_val_check = 0

        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())


            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss, acc, = self._train_batch_data(imgs, labels)
                print("lossss ====, acc===", loss, acc )
                # running_loss_cls += loss_cls
                # running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1
                # print("losss ====", running_loss / self.board_loss_every, "====accc====", running_acc)
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss', loss_board, self.step)
                    acc_board = running_acc / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Acc', acc_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(
                        'Training/Learning_rate', lr, self.step)
                    # loss_cls_board = running_loss_cls / self.board_loss_every
                    # self.writer.add_scalar(
                    #     'Training/Loss_cls', loss_cls_board, self.step)
                    # loss_ft_board = running_loss_ft / self.board_loss_every
                    # self.writer.add_scalar(
                    #     'Training/Loss_ft', loss_ft_board, self.step)

                    running_loss = 0.
                    running_acc = 0.
                    # running_loss_cls = 0.
                    # running_loss_ft = 0.
                if self.step % self.save_every == 0 and self.step != 0:
                    time_stamp = get_time()
                    self._save_state(time_stamp, extra=self.conf.job_name)
          
  
            print("start evaluate=======")
            for sample, ft_sample, target in tqdm(iter(self.val_loader)):
                imgs = [sample, ft_sample]
                labels = target

                loss_val, acc_val, = self.eva_val(imgs, labels)
                acc_val = acc_val.cpu().numpy()[0]

                print("lossss _ val ====, acc_val ===", loss_val, acc_val )

            if loss_val_check /2 <= acc_val:
                loss_val_check = acc_val
                time_stamp = get_time()
                self._save_state(time_stamp, extra=str(acc_val))

            self.schedule_lr.step()

        time_stamp = get_time()
        self._save_state(time_stamp, extra=self.conf.job_name)
        self.writer.close()

    def eva_val(self, imgs, labels):
        self.optimizer.zero_grad()
        device = torch.device('cuda:0')

        labels = labels.to(device)

        embeddings = self.model.forward(imgs[0].to(device))

        loss_cls = self.cls_criterion(embeddings, labels)
        # loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = loss_cls 
        acc = self._get_accuracy(embeddings, labels)[0]

        return loss.item(), acc

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        device = torch.device('cuda:0')

        labels = labels.to(device)

        embeddings = self.model.forward(imgs[0].to(device))

        loss_cls = self.cls_criterion(embeddings, labels)
        # loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = loss_cls 
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc


    def _define_network(self):
        model_path = "./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
      # load model weight
        model_name = os.path.basename(model_path)
        device = 'cuda'

        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input,)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

        state_dict = torch.load(model_path, map_location= device)

        keys = iter(state_dict)

        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        model = model.to('cuda')
        print("load done model ========")
        # frezee model
        i = 0
        for param in model.parameters():
            i += 1
            if i < 172:
                param.requires_grad = False
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))
        # torch.save( self.model , save_path + '/' +
        #            ('{}_{}_Verson2_model_iter-{}.pth'.format(time_stamp, extra, self.step)))

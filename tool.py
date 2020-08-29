from ast import NameConstant
from os import name
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from matplotlib.pyplot import axes, title, ylabel
from torch import mode, reshape, select, t
from torch.serialization import validate_cuda_device
import shutil
import datetime


def data_split(dataset, propertion=0.8, shuffle=True, rand=10):
    num = len(dataset)
    a = int(num*propertion)
    if shuffle == True:
        random.seed(rand)
        random.shuffle(dataset)
    return dataset[:a], dataset[a:]


class learning_curve():
    def __init__(self, records, labels=None, xlabel='', ylabel='', title='', reshape=False):
        if reshape == True:
            records = np.array(records).T
        self.records = records
        self.records_num = len(records)
        if labels is None:
            labels = list(range(1, len(self.records)+1))
        self.labels = labels
        self.select = [True for i in range(self.records_num)]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vlines = []
        self.hlines = []
        self.title = title
    def __getitem__(self,index):
        return self.records[index],self.labels[index]
    def add_curve(self,curves):
        for r,l in curves:
            self.records_num+=1
            self.records.append(r)
            self.labels.append(l)
            self.select.append(True)
    def __fix_conponents__(self, axe, legend=True, xlabel=None, ylabel=None, title=None):
        if legend == True:
            axe.legend()
        if xlabel is None:
            xlabel = self.xlabel
        if ylabel is None:
            ylabel = self.ylabel
        axe.set_xlabel(xlabel)
        axe.set_ylabel(ylabel)
        if title is None:
            title = self.title
        axe.set_title(title)

    def add_vlines(self, lines):
        self.vlines += lines
        self.vlines.sort()

    def reset_vlines(self):
        self.vlines = []

    def add_hlines(self, lines):
        self.hlines += lines
        self.hlines.sort()

    def reset_vlines(self):
        self.hlines = []

    def __draw_lines__(self, axe, a, b):
        for line in self.vlines:
            if line <= a:
                continue
            elif line >= b:
                break
            else:
                axe.axvline(line)
        for line in self.hlines:
            axe.axhline(line)

    def draw(self, a=0, b=-1, select=None):
        if b == -1:
            b = len(self.records[0])
        if select == None:
            select = [True for i in range(self.records_num)]
        fig, axe = plt.subplots()
        for i in range(self.records_num):
            if select[i] == True:
                axe.plot(
                    range(a, b), self.records[i][a:b], label=self.labels[i])
        self.__fix_conponents__(axe)
        self.__draw_lines__(axe, a, b)
        return fig

    def draw_each(self, a=0, b=-1, select=None):
        if b == -1:
            b = len(self.records[0])
        if select is None:
            select = self.select
        fig = plt.figure()
        k = 1
        for i in range(self.records_num):
            if select[i] == True:
                # plt.subplot(self.records_num,1,i+1)
                # plt.plot(range(a,b),self.records[i][a:b])
                # print(i)
                ax = fig.add_subplot(self.records_num, 1, k)
                k += 1
                ax.plot(range(a, b), self.records[i]
                        [a:b], label=self.labels[i])
                self.__fix_conponents__(ax)
                self.__draw_lines__(ax, a, b)
        fig.tight_layout()
        return fig

    def save(self, name, select=None):
        if select is None:
            select = self.select
        res = []
        for i in range(self.records_num):
            if select[i] == True:
                res.append(self.records[i])
        res2 = []
        res2.append(res)
        res2.append(self.hlines)
        res2.append(self.vlines)
        res = np.array(res)
        np.save(name, res2)

    def draw_range(self, a=0, b=-1, block_num=1, select=None, expand=0, each=False):
        if b == -1:
            b = len(self.records[0])
        if select is None:
            select = self.select
        block = int((b-a)/block_num)
        fig = []
        for i in range(block_num):
            a2 = i*block
            b2 = (i+1)*block
            a2 = a2-expand if a2-expand >= 0 else a2
            b2 = b2+expand if b2+expand <= b else b2
            print(i)
            if each is False:
                fig.append(self.draw(a2, b2, select=select))
            else:
                fig.append(self.draw_each(a2, b2, select))
            plt.show()
        return fig

    @staticmethod
    def load(name, labels=None, xlabel='', ylabel='', title=''):
        res = np.load(name)
        return learning_curve(res, labels, xlabel, ylabel, title)

    @staticmethod
    def load2(name, labels=None, xlabel='', ylabel='', title=''):
        res = np.load(name, allow_pickle=True)
        lc = learning_curve(res[0], labels, xlabel, ylabel, title)
        lc.add_hlines(list(res[1]))
        lc.add_vlines(list(res[2]))
        return lc


class ProgressBar:
    def __init__(self, num=100, info='ing...'):
        self.num = num
        self.curent = 0
        self.info = info
        pass

    def step(self, go=1):
        sys.stdout.write('\r')
        self.curent += go
        sys.stdout.write(f'{self.info} {self.curent/self.num*100:.2f}%')
        if self.curent >= self.num:
            sys.stdout.write('\r')

    def reset(self):
        self.curent = 0


class TrainTool():

    @staticmethod
    def train_once(model, train_data, lr=0.1, batch_size=10, epoch=10, optim=None):
        model.train()
        loader = data.DataLoader(train_data, batch_size=batch_size)
        if optim is None:
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        model.reset_loss_record()
        pb = ProgressBar(epoch*len(train_data), 'training...')
        for i in range(epoch):
            for x, y in loader:
                optim.zero_grad()
                res = model(x)
                loss = model.computeloss(x, y, res)
                loss.backward()
                optim.step()
                pb.step(len(x))
        return model.get_loss()

    @staticmethod
    def valid(model, valid_data, batch_size=100):
        loader = data.DataLoader(valid_data, batch_size=batch_size)
        model.reset_loss_record()
        pb = ProgressBar(len(valid_data), 'validing...')
        with torch.no_grad():
            for x, y in loader:
                res = model(x)
                loss = model.computeloss(x, y, res)
                pb.step(len(x))
        return model.get_loss()

    @staticmethod
    def train_epoch(model, train_data, valid_data, records=None, lr=0.1, batch_size=10, epoch=10, epoch_per=10, optim=None):
        if optim is None:
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_record = []
        for i in range(epoch):
            current = []
            current += TrainTool.train_once(model, train_data, lr=lr,
                                            batch_size=batch_size, epoch=epoch_per, optim=optim)
            current += (TrainTool.valid(model, valid_data,
                                        batch_size=batch_size))
            print(f'{i}', end='\t')
            print_list(current)
            loss_record.append((current))
            if records is not None:
                records.append(current)
        return loss_record

    @staticmethod
    def dlr(index): return 0.1

    @staticmethod
    def is_early_end(records):
        return False

    @staticmethod
    def train_adam_reset(model, train_data, valid_data, records=None, ex_records=None, dlr=dlr, batch_size=10, epoch=10, epoch_per=10, optim=None, early_end=is_early_end):
        loss_record = []
        reset_record = []
        last = 0
        reset_time = 0
        if optim is None:
            optim = torch.optim.Adam(model.parameters(), lr=dlr(reset_time))

        for i in range(epoch):
            current = []
            current += TrainTool.train_once(model, train_data, lr=0,
                                            batch_size=batch_size, epoch=epoch_per, optim=optim)
            current += (TrainTool.valid(model, valid_data,
                                        batch_size=batch_size))
            print(f'{i}', end='\t')
            print_list(current)
            loss_record.append((current))
            if records is not None:
                records.append(current)

            if early_end(loss_record, last):
                print('reset_adam')
                reset_time += 1
                optim = torch.optim.Adam(
                    model.parameters(), lr=dlr(reset_time))
                last = i
                reset_record.append(last)
                if ex_records is not None:
                    ex_records.append(last)
        return loss_record, reset_record

    @staticmethod
    def train_expand(model, train_data, valid_data, records=None, ex_records=None, dlr=dlr, batch_size=10, epoch=10, epoch_per=10, optim=None, early_end=is_early_end):
        loss_record = []
        reset_record = []
        last = 0
        reset_time = 0
        if optim is None:
            optim = torch.optim.Adam(model.parameters(), lr=dlr(reset_time))

        for i in range(epoch):
            current = []
            current += TrainTool.train_once(model, train_data, lr=0,
                                            batch_size=batch_size, epoch=epoch_per, optim=optim)
            current += (TrainTool.valid(model, valid_data,
                                        batch_size=batch_size))
            print(f'{i}', end='\t')
            print_list(current)
            loss_record.append((current))
            if records is not None:
                records.append(current)

            if early_end(loss_record, last):
                model.expand()
                print(model)
                reset_time += 1
                optim = torch.optim.Adam(
                    model.parameters(), lr=dlr(reset_time))
                last = i
                reset_record.append(last)
                if ex_records is not None:
                    ex_records.append(last)
        return loss_record, reset_record


def print_list(lst):
    for i in lst:
        print(f"{i:.8f}", end='\t')
    print('\n')   

def now_str():
    import datetime
    tz=datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz=tz).strftime('%Y_%m_%d_%H_%M_%S')

class colab_tool():

    @staticmethod
    def downloads(names):
        from google.colab import files
        for n in names:
            files.download(n)

    @staticmethod
    def download_drive(names):
        drive_path = 'drive/My Drive/'
        for n in names:
            tmp=drive_path+now_str()+'_'+n
            shutil.copy(n,tmp)

    @staticmethod
    def zip(zip_name,content):
        import shutil
        # Create 'path\to\zip_file.zip'
        shutil.make_archive(zip_name, 'zip', content)
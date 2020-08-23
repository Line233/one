from matplotlib.pyplot import axes, ylabel
import torch
import random
from torch import select
import torch.nn as nn
from torch.serialization import validate_cuda_device
import torchvision
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch.utils.data as data
import sys


def data_split(dataset,propertion=0.8,shuffle=True,rand=10):
    num=len(dataset)
    a=int(num*propertion)
    if shuffle == True:
        random.seed(rand)
        random.shuffle(dataset)
    return dataset[:a],dataset[a:]
    
class learning_curve():
    def __init__(self,records,labels=None,xlabel='',ylabel=''):
        self.records=records
        self.max_len=len(records[0])
        self.records_num=len(records)
        if labels is None:
            labels=range(1,len(self.records)+1)
        self.labels=labels
        self.select=[True for  i in range(self.records_num)]
        self.xlabel=xlabel
        self.ylabel=ylabel

    def __fix_conponents__(self,axe,legend=True,xlabel=None,ylabel=None,title=None):
        if legend==True:
            axe.legend()
        if xlabel is  None:
            xlabel=self.xlabel
        if ylabel is  None:
            ylabel=self.ylabel
        axe.set_xlabel(xlabel)
        axe.set_ylabel(ylabel)
        if title is not None:
            axe.set_title(title)
        

    def draw(self,a=0,b=-1,select=None):
        if b==-1:
            b=len(self.records[0])
        if select==None:
            select=[True for i in range(self.records_num)]
        fig,axe=plt.subplots()
        for i in range(self.records_num):
            if select[i] is True:
                axe.plot(range(a,b),self.records[i][a:b],label=self.labels[i])
        self.__fix_conponents__(axe)
        return fig
        
    def draw_each(self,a=0,b=-1,select=None):
        if b==-1:
            b=len(self.records[0])
        if select is None:
            select=self.select
        fig=plt.figure()
        k=1
        for i in range(self.records_num):
            if select[i] is True:
                # plt.subplot(self.records_num,1,i+1)
                # plt.plot(range(a,b),self.records[i][a:b])
                # print(i)
                ax=fig.add_subplot(self.records_num,1,k)
                k+=1
                ax.plot(range(a,b),self.records[i][a:b],label=self.labels[i])
                self.__fix_conponents__(ax)
        fig.tight_layout()
        return fig
    def save(self,name,select=None):
        if select is None:
            select=self.select
        res=[]
        for i in range(self.records_num):
            if select[i] is True:
                res.append(self.records[i])
        res=np.array(res)
        np.save(name,res)
    
    def draw_range(self,a=0,b=-1,block_num=1,select=None,expand=0,each=False):
        if b==-1:
            b=self.max_len
        if select is None:
            select=self.select
        block=int((b-a)/block_num)
        fig=[]
        for i in range(block_num):
            a2=i*block
            b2=(i+1)*block
            a2=a2-expand if a2-expand>=0 else a2
            b2=b2+expand if b2+expand<=b else b2
            print(i)
            if each is False:
                fig.append(self.draw(a2,b2,select=select))
            else:
                fig.append(self.draw_each(a2,b2,select))
            plt.show()
        return fig
        
    @staticmethod
    def load(name,labels=None,xlabel='',ylabel=''):
        res=np.load(name)
        return learning_curve(res,labels,xlabel,ylabel)

class ProgressBar:
    def __init__(self,num=100,info='ing...'):
        self.num=num
        self.curent=0
        self.info=info
        pass
    def step(self,go=1):
        sys.stdout.write('\r')
        self.curent+=go
        sys.stdout.write(f'{self.info} {self.curent/self.num*100:.2f}%')
        if self.curent>=self.num:
            sys.stdout.write('\r')
    def reset(self):
        self.curent=0

class TrainTool():
    
    @staticmethod
    def train_once(model,train_data,lr=0.1,batch_size=10,epoch=10,optim=None):
        model.train()
        loader=data.DataLoader(train_data,batch_size=batch_size)
        if optim is None:
            optim=torch.optim.Adam(model.parameters(),lr=lr)
        model.reset_loss_record()
        pb=ProgressBar(epoch*len(train_data),'training...')
        for i in range(epoch):
            for x,y in loader:
                optim.zero_grad()
                res=model(x)
                loss=model.computeloss(x,y,res)
                loss.backward()
                optim.step()
                pb.step(len(x))
        return model.get_loss()
    @staticmethod
    def valid(model,valid_data,batch_size=100):
        loader=data.DataLoader(valid_data,batch_size=batch_size)
        model.reset_loss_record()
        pb=ProgressBar(len(valid_data),'validing...')
        with torch.no_grad():
            for x,y in loader:
                res=model(x)
                loss=model.computeloss(x,y,res)
                pb.step(len(x))
        return model.get_loss()
        
    @staticmethod
    def train_epoch(model,train_data,valid_data,lr=0.1,batch_size=10,epoch=10,epoch_per=10,optim=None):
        if optim is None:
            optim=torch.optim.Adam(model.parameters(),lr=lr)
        loss_record=[]
        valid_record=[]
        for i in range(epoch):
            loss_record.append(TrainTool.train_once(model,train_data,lr=lr,batch_size=batch_size,epoch=epoch_per,optim=optim))
            valid_record.append(TrainTool.valid(model,valid_data,batch_size=batch_size))
            print(f'{i}\t{loss_record[-1]}\t{valid_record[-1]}')
        return loss_record,valid_record
from matplotlib.pyplot import axes, ylabel
import torch
import random
from torch import select
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
    def __init__(self,num=100):
        self.num=num
        self.curent=0
        pass
    def step(self,go=1):
        sys.stdout.write('\r')
        self.curent+=go
        sys.stdout.write(f'{self.curent/self.num*100:.2f}%')
    def reset(self):
        self.curent=0
    def end(self):
        sys.stdout.write('\r')



        
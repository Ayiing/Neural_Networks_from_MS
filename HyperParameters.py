from enum import Enum
import numpy as np
class NetType(Enum):
    Fitting=1
    BinaryClassifier=2
    BinaryClassifier_tanh=3
    MultipleClassifier=4
class HyperParameters_1_0(object):
    def __init__(self,eta=0.1,max_epoch=1000,batch_size=5,eps=0.1,net_type=NetType.Fitting):
        self.eta=eta
        self.max_epoch=max_epoch#最大训练周期
        self.batch_size=batch_size#一批多少个训练样本
        self.eps=eps#最后误差
        self.net_type=net_type
class Logistic(object):
    def forward(self,z):
        a=1.0/(1.0+np.exp(-z))
        return a
class Tanh(object):
    def forward(self,z):
        a=2/(1+np.exp(-2*z))-1
        return a
class Softmax(object):
    def forward(self,z):
        '''
        e_x=np.exp(z)
        A=e_x/np.sum(e_x,exis=1,keepdims=True)
        '''
        shift_z=z-np.max(z,axis=1,keepdims=True)#找到一行中最大的数
        exp_z=np.exp(shift_z)
        a=exp_z/np.sum(exp_z,axis=1,keepdims=True)#如果不用keepdims，np.sum就变成一维数组了
        return a
class LossFunction(object):
    def __init__(self,net_type,e=1e-10):
        self.net_type=net_type#1,2,3,4
        self.e=e#log(e+..)保证交叉熵loss不要出现nan
    def CheckLoss(self,A,Y):
        m=A.shape[0]
        if self.net_type == NetType.BinaryClassifier:
            loss=self.CE2(A,Y,m)
        elif self.net_type == NetType.BinaryClassifier_tanh:
            loss=self.CE2_tanh(A,Y,m)
        elif self.net_type == NetType.MultipleClassifier:
            loss=self.CE3(A,Y,m)
        else:
            loss=self.MSE(A,Y,m)
        return loss
    def MSE(self,A,Y,count):
        dz=A-Y
        loss=(dz**2).sum()/count/2#仍认为一个输出
        return loss
    def CE2(self,A,Y,count):
        #loss=-[ylna+(1-y)ln(1-a)]
        p1=1-Y
        p2=np.log(1-A+self.e)
        p3=np.log(A+self.e)
        p4=np.multiply(p1,p2)#对应元素相乘
        p5=np.multiply(Y,p3)
        Loss=np.sum(-(p4+p5))
        loss=Loss/count
        return loss
    def CE2_tanh(self,A,Y,count):
        #loss=-[(1+y)ln((1+a)/2)+(1-y)ln((1-a)/2)]
        p1=1+Y
        p2=np.log((1+A)/2+self.e)
        p3=1-Y
        p4=np.log((1-A)/2+self.e)
        p5=np.multiply(p1,p2)
        p6=np.multiply(p3,p4)
        Loss=np.sum(-(p5+p6))
        loss=Loss/count
        return loss
    def CE3(self,A,Y,count):
        p1=np.log(A)
        p2=np.multiply(Y,p1)#对应元素相乘
        Loss=np.sum(-p2)
        loss=Loss/count
        return loss
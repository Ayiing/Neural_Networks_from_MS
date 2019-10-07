import numpy as np
from LoadData import DataReader_1_0
from HyperParameters import *
import matplotlib.pyplot as plt
class NeuralNet(object):
    def __init__(self,params,input_size,output_size):
        self.params=params
        self.W=np.zeros([input_size,output_size])
        self.B=np.zeros([1,output_size])
    def __forward(self,batch_x):
        batch_z=np.dot(batch_x,self.W)+self.B
        if self.params.net_type == NetType.BinaryClassifier:
            A=Logistic().forward(batch_z)#
            return A
        elif self.params.net_type == NetType.BinaryClassifier_tanh:
            A=Tanh().forward(batch_z)
            return A
        elif self.params.net_type == NetType.MultipleClassifier:
            A=Softmax().forward(batch_z)
            return A
        else:
            return batch_z
    def __backward(self,batch_x,batch_y,batch_a):
        m=batch_x.shape[0]
        if self.params.net_type == NetType.BinaryClassifier_tanh:
            dz=2*(batch_a-batch_y)#(batch_a-batch_y)*(1+batch_a)/batch_a
        else:
            dz=batch_a-batch_y#loss对z的导数
        dW=np.dot(batch_x.T,dz)/m#dW=np.dot(x.T,dz) 一个样本
        dB=dz.sum(axis=0,keepdims=True)/m#dB=dz?keepdims 必须吗？,必须 小心退化成一维数组,多个神经元了
        return dW,dB
    def __update(self,dW,dB):
        self.W=self.W-self.params.eta*dW
        self.B=self.B-self.params.eta*dB
    def __checkloss(self,dataReader):
        X,Y=dataReader.GetWholeTrainSamples()#XTrain,YTrain
        A=self.__forward(X)
        Loss=LossFunction(self.params.net_type)
        loss=Loss.CheckLoss(A,Y)
        return loss
    def inference(self,x):
        return self.__forward(x)
    def Train(self,readData,checkpoint=0.1):
        if self.params.batch_size<0:
            self.params.batch_size=readData.num_train
        m=int(readData.num_train/self.params.batch_size)
        checkpoint_iteration=int(m*checkpoint)#防止m不是整十的倍数，导致checkpoint不是整数
        iteration=[]
        Loss=[]
        loss=10
        for epoch in range(self.params.max_epoch):
            readData.Shuffle()
            for i in range(m):
                batch_x,batch_y=readData.GetBatchTrainSamples(self.params.batch_size,i)
                batch_z=self.__forward(batch_x)
                dW,dB=self.__backward(batch_x,batch_y,batch_z)
                self.__update(dW,dB)
                loss=self.__checkloss(readData)
                if (i+epoch*m+1)%checkpoint_iteration==0:
                    #画图看一下iteration 和loss的变化
                    #if(self.params.eta>1e-4):
                    #    self.params.eta=self.params.eta*0.1
                    iteration.append(i+epoch*m)
                    Loss.append(loss)
                    print(epoch,"i:",i,"w:\n",self.W,"b:\n",self.B,"loss:",loss)
                if loss<self.params.eps:
                    break
            if loss<self.params.eps:#loss出了最内的for循环，依然健在
                break
        print("result:\nw:\n",self.W,"b:\n",self.B,"loss:",loss)
        self.ShowLoss(iteration,Loss)
    def ShowLoss(self,iteration,Loss):
        plt.plot(iteration,Loss)
        plt.show()
def draw(net,reader):
    #先把所有散点画出来
    #X,Y=reader.GetWholeTrainSamples()
    X=reader.XRaw
    Y=reader.YRaw
    '''
    plt.plot(X[:,0],X[:,1],'b.')
    x=np.linspace(1,5,100).reshape(50,2)
    y=net.inference(x)
    plt.plot(x,y)
    '''
    for i in range(reader.num_train):
        if Y[i,0]==1:
            plt.scatter(X[i,0],X[i,1],marker='x',c='g')
        elif Y[i,0]==2:
            plt.scatter(X[i,0],X[i,1],marker='o',c='r')
        else:
            plt.scatter(X[i,0],X[i,1],marker='v',c='k')
    #画分界线
    w=-net.W[0,0]/net.W[1,0]
    b=-net.B[0,0]/net.W[1,0]
    x=np.linspace(0,1)
    y=w*x+b
    plt.plot(x,y)
    #plt.show()
def draw_predicate(x):
    for i in x:
        plt.scatter(i[0],i[1],marker='*',c='b')
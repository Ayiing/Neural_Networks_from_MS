import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
class HyperParameters_1_0(object):
    def __init__(self,input_size,output_size,eta=0.1,max_epoch=1000,batch_size=5,eps=0.1):
        self.input_size=input_size
        self.output_size=output_size
        self.eta=eta
        self.max_epoch=max_epoch#最大训练周期
        self.batch_size=batch_size#一批多少个训练样本
        self.eps=eps#最后误差
class DataReader_1_0(object):
    def __init__(self,data_file):
        self.train_file_name=data_file
        self.num_train=0
        self.XTrain=None
        self.YTrain=None
        self.XRaw=None#由于需要归一化，用以保留原数据
        self.YRaw=None
    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data=np.load(self.train_file_name)
            self.XRaw=data["data"]
            self.YRaw=data["label"]
            self.num_train=self.XRaw.shape[0]#shape[0]行数
            self.XTrain=self.XRaw
            self.YTrain=self.YRaw
        else:
            raise Exception("cant find train file!")
    def GetSingleTrainSample(self,iteration):
        x=self.XTrain[iteration]
        y=self.YTrain[iteration]
        return x,y
    def GetBatchTrainSamples(self,batch_size,iteration):
        start=iteration*batch_size#iteration 从0开始
        end=start+batch_size
        batch_x=self.XTrain[start:end,:]# 把start-end的样本 所有特征都取出来
        batch_y=self.YTrain[start:end,:]
        return batch_x,batch_y
    def GetWholeTrainSamples(self):
        return self.XTrain,self.YTrain
    def Shuffle(self):
        #shuffle:洗牌，对XTrain和YTrain进行洗牌重新排列
        seed=np.random.randint(0,100)
        np.random.seed(seed)
        XP=np.random.permutation(self.XTrain)
        np.random.seed(seed)#注意保持XP和YP的对应
        YP=np.random.permutation(self.YTrain)
        self.XTrain=XP
        self.YTrain=YP
    def NormalizeX(self,split=False):
        #min-max归一化
        '''
        Xmin=self.XTrain.min(axis=0)
        Xmax=self.XTrain.max(axis=0)
        self.XTrain=(self.XTrain-Xmin)/(Xmax-Xmin)
        '''
        #由于xraw被分为train和test部分，分别对其归一hua
        X_newTrain=np.zeros(self.XTrain.shape)
        num_feature=self.XTrain.shape[1]#特征个数
        self.X_norm=np.zeros((num_feature,2))#一列保存min 一列保存max-min
        for i in range(num_feature):
            col_i=self.XRaw[:,i]#所有行第i列,而且退化成一维数组，不再是列向量
            col_iTrain=self.XTrain[:,i]
            min_value=np.min(col_i)
            max_value=np.max(col_i)
            self.X_norm[i,0]=min_value
            self.X_norm[i,1]=max_value-min_value
            new_col=(col_iTrain-self.X_norm[i,0])/self.X_norm[i,1]
            X_newTrain[:,i]=new_col
        #print("Self.X_norm:",self.X_norm)
        self.XTrain=X_newTrain
        if split:
            X_newTest=np.zeros(self.XTest.shape)
            for i in range(num_feature):
                col_iTest=self.XTest[:,i]
                new_col=(col_iTest-self.X_norm[i,0])/self.X_norm[i,1]
                X_newTest[:,i]=new_col
            self.XTest=X_newTest
        #平均值归一化
        #非线性归一化
    def NormalizeY(self,split=False):
        #由于X归一化之后，Loss依然很大，是因为和Y在一个数量级上,所以把y也归一化试试
        self.Y_norm=np.zeros((1,2))
        max_value=np.max(self.YRaw)
        min_value=np.min(self.YRaw)
        self.Y_norm[0,0]=min_value
        self.Y_norm[0,1]=max_value-min_value
        Y_newTrain=(self.YTrain-self.Y_norm[0,0])/self.Y_norm[0,1]
        if split:
            Y_newTest=(self.YTest-self.Y_norm[0,0])/self.Y_norm[0,1]
            self.YTest=Y_newTest
        self.YTrain=Y_newTrain
    def DenormalizeY(self,y):
        #把给的y变成原来的real数据
        return y*self.Y_norm[0,1]+self.Y_norm[0,0]
    def NormalizePredicateData(self,x):
        num_feature=x.shape[1]
        X_new=np.zeros(x.shape)
        for i in range(num_feature):
            col_i=x[:,i]
            new_col=(col_i-self.X_norm[i,0])/self.X_norm[i,1]
            X_new[:,i]=new_col
        return X_new
    def ToZeroOne(self):
        #为了配合tanh作为分类函数a，把标签值改为【-1，1】
        Y=np.zeros(self.YTrain.shape)
        for i in range(self.num_train):
            if self.YTrain[i,0]==0:
                Y[i,0]=-1
            elif self.YTrain[i,0]==1:
                Y[i,0]=1
        self.YTrain=Y
        self.YRaw=Y
    def ToOneHot(self,num_category,base=0,split=False):
        count=self.YTrain.shape[0]
        self.num_category=num_category
        y_new=np.zeros((count,self.num_category))
        for i in range(count):
            n=(int)(self.YTrain[i,0])
            y_new[i,n-base]=1#注意啊，base应该=1
        self.YTrain=y_new
        if split:
            count=self.YTest.shape[0]
            y_new=np.zeros((count,self.num_category))
            for i in range(count):
                n=int(self.YTest[i,0])
                y_new[i,n-base]=1
            self.YTest=y_new
def draw(net,dataReader):
    X,Y=dataReader.GetWholeTrainSamples()
    plt.plot(X,Y,"b.")#画出散点
    x=np.linspace(0,1,100)#生成的是1*n shape(100,)没有shape[1]是一个一维数组[1,2,3]，但是[[1,2,3]]才能转置，shape（1，1）是一个矩阵，一行一列只有一个元素[[x]]
    y=net.inference(x.reshape(100,1))
    plt.plot(x,y)
    plt.show()
'''
a=np.array([1,2,3,4]).reshape(4,1)
print(a*(-1)+2)
a=np.array([ [1,2],[-2,5],[3,-1] ])
b=a[:,1]
c=np.zeros((3,2))
print(b)#行
c[:,1]=b#又可以转变回去了
print(c)
print((b-np.min(b))/(np.max(b)-np.min(b)))
print(b)
print(a[1,1])
d=np.array([[1,-1]])
print(a/d)
print(a[:,1])#所有行的1列,但是转化成行向量了，一维数组了
print(a.shape)
b=a.min(axis=0)#每一列最小值
print(a-b)
c=a.mean(axis=0)
print(c,a-c)

seed=5
np.random.seed(seed)
a=[4,1,2,3]
e=[1,0,1,0]
b=np.random.permutation(a)
np.random.seed(seed)
c=np.random.permutation(e)
print(a,b,c)
'''
'''
a=np.array([[1,2,3]])
print(type(a))
b=np.linspace(-1,1,5)#b=[1,2,3] 是个一维数组，不是矩阵，不能用.T 可以用reshape
print(type(b))
print(b)
print(b.T)
print(b.reshape(5,1))
print(a.shape[0],a.shape[1],b.shape[0])#,b.shape[1])
b=np.array([[1],[2],[3]])
'''
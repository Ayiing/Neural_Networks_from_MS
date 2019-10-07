from NeuralNet import *
import pandas as pd
from pathlib import Path
class DataReader_iris(DataReader_1_0):
    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data=pd.read_csv(train_file).values
            self.XRaw=data[:,:4]
            self.YRaw=data[:,4:]
            self.num_train=self.XRaw.shape[0]
            self.XTrain=self.XRaw
            self.YTrain=self.YRaw
        else:
            raise Exception('cant open the train file!!')
    def SplitData(self,split_ratio=0.5):
        total_len=self.XRaw.shape[0]
        self.num_train=int(total_len*split_ratio)
        self.Shuffle()#把Train（这时候和Raw相等）训练集洗牌一遍
        #把原始数据划分为训练集和测试集，
        self.XTest=self.XTrain[self.num_train:,:]#包前不包后，
        self.YTest=self.YTrain[self.num_train:,:]
        self.XTrain=self.XTrain[0:self.num_train,:]#前000行做训练集
        self.YTrain=self.YTrain[0:self.num_train,:]
        #print(self.XRaw.shape,self.XTrain.shape,self.XTest.shape)
def varify(net,reader):
    #用test部分检验一下正确率
    X,Y=reader.XTest,reader.YTest
    count=0
    total=Y.shape[0]
    for i in range(total):
        Z=net.inference(X[i])
        #经过softmax得到的Z是一个 三个概率
        z=np.argmax(Z)+1
        y=np.argmax(Y[i])+1
        #print(z,y)
        if z==y:
            count+=1
    print("ratio_correct:",count*1.0/total)
if __name__=='__main__':
    file_name='F:/pyzo/network/data/iris.csv'
    #读取数据
    reader=DataReader_iris(file_name)
    reader.ReadData()
    reader.SplitData(split_ratio=0.5)
    reader.NormalizeX(split=True)
    num_category=3#0,1,2
    reader.ToOneHot(num_category,base=1,split=True)
    #(reader.YTest)
    #创建神经元
    params=HyperParameters_1_0(eta=0.1,max_epoch=500,batch_size=10,eps=1e-3,net_type=NetType.MultipleClassifier)
    input=4
    output=3
    net=NeuralNet(params,input,output)
    #开始训练
    net.Train(reader,checkpoint=1)
    varify(net,reader)

from NeuralNet import *
from pathlib import Path
import pandas as pd
class DataReader_boston(DataReader_1_0):
    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data=pd.read_csv(train_file)
            head=['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']#自变量
            self.XRaw=data[head].values#dataframe取values得到出了列名以外的数值，转化成ndarray了
            self.YRaw=data['medv'].values.reshape(506,1)
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
def show_result(net,reader):
    X,Y=reader.GetWholeTrainSamples()
    num_feature=X.shape[1]#特征个数
    n=X.shape[0]
    head=['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat']#自变量
    for i in range(num_feature):
        #每一个特征分别和Y作图
        for j in range(n):
            plt.scatter(X[j,i],Y[j,0],marker='*',c='r')
        x=np.linspace(0,1,50)
        y=[]
        for k in x:
            y.append(k*net.W[i,0]+net.B[0,0])
        plt.plot(x,y)
        plt.xlabel(head[i])
        plt.show()
def varify(net,reader):
    #横坐标是真实结果，纵坐标是预测结果
    plt.xlabel('real_result')
    plt.ylabel('predicate_result')
    X,Y=reader.XTest,reader.YTest
    Z=net.inference(X)
    #把Y，Z去归一化操作
    Y=reader.DenormalizeY(Y)
    Z=reader.DenormalizeY(Z)
    '''
    for i in range(X.shape[0]):
        plt.scatter(Y[i,0],Z[i,0],marker='^',c='r')
    #画一条45'的线
    x=np.linspace(0,51,50)
    y=x
    plt.plot(x,y)
    '''
    #那我也可以把Y-Z给画出来，顺便设定一个eps，abs<eps即认为相等
    correct=0
    total=X.shape[0]
    for i in range(total):
        diff=(Y[i,0]-Z[i,0])/Y[i,0]
        eps=0.1#10%的误差
        if abs(diff)<=eps:
            plt.scatter(i,0,marker='^',c='r')
            correct+=1
        else:
            plt.scatter(i,diff,marker='*',c='b')
    print("正确率：",correct*1.0/total)
    plt.show()
def Test(net,reader):
    XTest=reader.XTest
    Y=reader.YTest
    A=net.inference(XTest)
    LOSS=LossFunction(NetType.Fitting,e=0)#e是防止交叉熵变得很大，但这里用MSE均方差
    loss=LOSS.CheckLoss(A,Y)
    print("loss:",loss)
if __name__=='__main__':
    file_name='F:/pyzo/network/data/boston_housing.csv'
    #读取数据
    reader=DataReader_boston(file_name)
    reader.ReadData()
    #虽然我先切分，后归一化，但实际上是我做的是：先把数据集整体归一化，再把数据集切分为测试集和训练集
    reader.SplitData(split_ratio=0.5)#切分为数据集和测试集
    reader.NormalizeX(split=True)#归一化特征,
    reader.NormalizeY(split=True)
    #神经元
    params=HyperParameters_1_0(eta=0.01,max_epoch=2000,batch_size=30,eps=1e-5,net_type=NetType.Fitting)
    input=13
    output=1
    net=NeuralNet(params,input,output)
    #开始训练
    net.Train(reader,checkpoint=1)
    varify(net,reader)
    #show_result(net,reader)
    #测试一下啊
    '''
    xt=np.array([0.04741,0,11.93,0,0.573,6.03,80.8,2.505,1,273,21,396.9,7.88]).reshape(1,13)
    xt=reader.NormalizePredicateData(xt)
    print(net.inference(xt))
    '''
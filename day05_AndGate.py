from NeuralNet import *
class ReadLogisticData(DataReader_1_0):
    def ReadData_AndGate(self):
        X=np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y=np.array([0,0,0,1]).reshape(4,1)
        self.XTrain=self.XRaw=X
        self.YTrain=self.YRaw=Y
        self.num_train=self.XTrain.shape[0]
    def ReadData_NandGate(self):
        #与非门，
        X=np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y=np.array([1,1,1,0]).reshape(4,1)
        self.XTrain=self.XRaw=X
        self.YTrain=self.YRaw=Y
        self.num_train=self.XTrain.shape[0]
    def ReadData_OrGate(self):
        X=np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y=np.array([0,1,1,1]).reshape(4,1)
        self.XTrain=self.XRaw=X
        self.YTrain=self.YRaw=Y
        self.num_train=self.XTrain.shape[0]
    def ReadData_NorGate(self):
        X=np.array([0,0,0,1,1,0,1,1]).reshape(4,2)
        Y=np.array([1,0,0,0]).reshape(4,1)
        self.XTrain=self.XRaw=X
        self.YTrain=self.YRaw=Y
        self.num_train=self.XTrain.shape[0]
def Test(net,reader):
    X,Y=reader.GetWholeTrainSamples()
    A=net.inference(X)
    diff=np.abs(A-Y)
    result=np.where(diff<1e-2,True,False)
    if result.sum()==4:#四个样本的误差都小于1e-2才算过
        return True
    else:
        return False
if __name__=="__main__":
    #读取数据
    file_name='F:\pyzo\network\data\ch06.npz'
    reader=ReadLogisticData(file_name)
    reader.ReadData_NandGate()
    #神经元
    params=HyperParameters_1_0(eta=0.5,max_epoch=5000,batch_size=1,eps=2e-3,net_type=NetType.BinaryClassifier)
    input=2
    output=1
    net=NeuralNet(params,input,output)
    #训练
    '''
    times=1
    while not Test(net,reader):
        net.Train(reader,checkpoint=1)
        times+=1
    print(Test(net,reader))
    print(times*net.params.max_epoch*reader.XTrain.shape[0])#迭代次数,也不准，因为满足eps跳出 而不是达到max_epoch退出的train
    '''
    net.Train(reader,checkpoint=10)
    #画图
    draw(net,reader)
    plt.show()
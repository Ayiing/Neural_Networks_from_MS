from LoadData import *
class LogicNotGateDataReader(DataReader_1_0):
    def ReadData(self):
        X=np.array([0,1]).reshape(2,1)
        Y=np.array([1,0]).reshape(2,1)
        self.XTrain=X
        self.YTrain=Y
        self.num_train=2
if __name__=="__main__":
    #读取数据
    #file_name='f:/pyzo/network/ch04.npz'
    sdr=LogicNotGateDataReader(None)
    sdr.ReadData()
    params=HyperParameters_1_0(1,1,eta=0.1,max_epoch=1000,batch_size=1,eps=1e-8)
    net=NeuralNet(params)
    net.Train(sdr)
    draw(net,sdr)
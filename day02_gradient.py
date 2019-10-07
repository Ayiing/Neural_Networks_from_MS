import numpy as np
from pathlib import Path
class DataReader(object):
    def __init__(self,data_file):
        self.train_file_name=data_file
        self.num_train=0
        self.XTrain=None
        self.YTrain=None
    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data=np.load(self.train_file_name)
            self.XTrain=data["data"]
            self.YTrain=data["label"]
            self.num_train=self.XTrain.shape[0]
        else:
            raise Exception("cant find train file!")
    def GetSingleTrainSample(self,iteration):
        x=self.XTrain[iteration]
        y=self.YTrain[iteration]
        return x,y
    def GetBatchTrainSamples(self,batch_size,iteration):
        start=iteration*batch_size
        end=start+batch_size
        bacth_x=self.XTrain[start:end,:]# 把start-end的样本 所有特征都取出来
        batch_y=self.YTrain[start:end,:]
        return batch_x,batch_y
    def GetWholeTrainSamples(self):
        return self.XTrain,self.YTrain
file_name="F:/pyzo/network/ch04.npz"
if __name__=="__main__":
    reader=DataReader(file_name)
    reader.ReadData()
    X,Y=reader.GetWholeTrainSamples()
    eta=0.1
    w,b=0.0,0.0#xi是一个数，所以w。b用的都是数,！标量！
    for i in range(reader.num_train):
        xi=X[i]
        yi=Y[i]
        zi=xi*w+b
        dz=zi-yi#loss=1/2 (zi-yi)^2 对z求导
        dw=dz*xi#loss对w求导，xi为z对w的求导
        db=dz*1#loss对b求导，1是z对b的求导
        print("xi:",xi,"yi:",yi,"zi:",zi,"w:",w,"b:",b)
        w=w-eta*dw
        b=b-eta*db
    print("w=:",w)
    print("b=:",b)
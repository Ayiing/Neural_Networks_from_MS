from LoadData import DataReader_1_0
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class NeuralNet(object):
    def __init__(self,eta):
        self.eta=eta
        self.w=0#针对这个问题，样本只有一个特征，w、b都是标量
        self.b=0
    def __forward(self,x):
        #向前计算,并且保持私有
        z=x*self.w+self.b
        return z
    def __backward(self,x,y,z):
        #向后计算,反向传播，调整w、b
        dz=z-y
        db=dz
        dw=x*dz
        return dw,db
    def __update(self,dw,db):
        #反向传播完之后，更新w、b
        self.w=self.w-self.eta*dw
        self.b=self.b-self.eta*db
    def train(self,dataReader):
        for i in range(dataReader.num_train):
            #只训练一轮
            x,y=dataReader.GetSingleTrainSample(i)
            z=self.__forward(x)
            dw,db=self.__backward(x,y,z)
            self.__update(dw,db)
            print(i,"w:",self.w,"b:",self.b)
    def inference(self,x):
        return self.__forward(x)
def draw(net,dataReader):
    X,Y=dataReader.GetWholeTrainSamples()
    plt.plot(X,Y,"b.")
    PX=np.linspace(0,1,10)
    PZ=net.inference(PX)#注意这里在带入 PZ=PX*self.w+self.b 计算的时候，的计算规则
    #print("PX:",PX,"PZ:",PZ)
    plt.plot(PX,PZ,"r")
    plt.show()
file_name="F:/pyzo/network/ch04.npz"
if __name__=='__main__':
    reader=DataReader_1_0(file_name)
    reader.ReadData()
    eta=0.1
    net=NeuralNet(eta)
    net.train(reader)
    print("w=%f,b=%f"%(net.w,net.b))
    result=net.inference(0.346)
    print("result=",result)
    draw(net,reader)

import numpy as np
from LoadData import DataReader_1_0,HyperParameters_1_0
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class NeuralNet(object):
    def __init__(self,params):
        self.params=params
        self.W=np.zeros([self.params.input_size,self.params.output_size])
        self.B=np.zeros([1,self.params.output_size])
    def __forward(self,batch_x):
        batch_z=np.dot(batch_x,self.W)+self.B
        return batch_z
    def __backward(self,batch_x,batch_y,batch_z):
        m=batch_x.shape[0]
        dz=batch_z-batch_y
        dW=np.dot(batch_x.T,dz)#dW=np.dot(x.T,dz) 一个样本
        dB=dz.sum(axis=0,keepdims=True)/m#dB=dz?keepdims 必须吗？
        return dW,dB
    def __update(self,dW,dB):
        self.W=self.W-self.params.eta*dW
        self.B=self.B-self.params.eta*dB
    def __checkloss(self,dataReader):
        X,Y=dataReader.GetWholeTrainSamples()
        m=X.shape[0]
        Z=self.__forward(X)
        Loss=((Z-Y)**2).sum(axis=0)/2/m
        return Loss
    def Train(self,readData):
        if self.params.batch_size<0:
            self.params.batch_size=readData.num_train
        m=int(readData.num_train/self.params.batch_size)
        iteration=[]
        Loss=[]
        for epoch in range(self.params.max_epoch):
            readData.Shuffle()
            for i in range(m):
                batch_x,batch_y=readData.GetBatchTrainSamples(self.params.batch_size,i)
                batch_z=self.__forward(batch_x)
                dW,dB=self.__backward(batch_x,batch_y,batch_z)
                self.__update(dW,dB)
                #dz=batch_z-batch_y
                #loss=(dz**2).sum()/(2*m)#我这样算的loss是针对当前batch_size个样本的loss
                loss=self.__checkloss(readData)
                if (i+1)%10==0:
                    #画图看一下iteration 和loss的变化
                    #if(self.params.eta>1e-4):
                    #    self.params.eta=self.params.eta*0.1
                    iteration.append(i+epoch*m)
                    Loss.append(loss)
                    print(epoch,"i:",i,"w:\n",self.W,"b:",self.B,"loss:",loss)
                if loss<self.params.eps:
                    break
            if loss<self.params.eps:#loss出了最内的for循环，依然健在
                break
        print("result:\nw:\n",self.W,"b:",self.B,"loss:",loss)
        draw_2D(iteration,Loss)
    def inference(self,x):
        return self.__forward(x)
    def draw_3D(self,readData):
        fig=plt.figure()
        ax=Axes3D(fig)
        x,y=readData.GetWholeTrainSamples()
        X1,X2=np.meshgrid(x.T[0],x.T[1])#(x[:,1],x[:,2])
        ax.plot_surface(X1,X2,y,cmap='rainbow')
        u=np.linspace(-3,3,100)
        v=np.linspace(-3,3,100)
        U,V=np.meshgrid(u,v)
        R=np.zeros([len(u),len(v)])
        for i in range(len(u)):
            for j in range(len(v)):
                R[i,j]=U[i,j]*self.W[0]+V[i,j]*self.W[1]+self.B
        ax.plot_surface(U,V,R,cmap='black')
        plt.show()
def DeNormalizeWeightsBias(net,dataReader):
    W_real=np.zeros(net.W.shape)
    num_feature=dataReader.XRaw.shape[1]#太坑了，刚开始用的Xtrain，XTrain已经被归一化了
    X_norm=np.zeros((num_feature,2))
    for i in range(num_feature):
        col_i=dataReader.XRaw[:,i]
        X_norm[i,0]=np.min(col_i)
        X_norm[i,1]=np.max(col_i)-np.min(col_i)
        W_real[i,0]=net.W[i,0]/X_norm[i,1]
    #B_real=net.B-net.W[0,0]*X_norm[0,0]/X_norm[0,1]-net.W[1,0]*X_norm[1,0]/X_norm[1,1]
    #print("X_norm:",X_norm)
    print("W_real:\n",W_real)
    B_real=net.B-W_real[0,0]*X_norm[0,0]-W_real[1,0]*X_norm[1,0]
    return W_real,B_real
def DeNormalizeY(Z,dataReader):
    Z_real=Z*dataReader.Y_norm[0,1]+dataReader.Y_norm[0,0]
    return Z_real
def draw_3D(net,dataReader):
    X,Y=reader.GetWholeTrainSamples()
    fig=plt.figure()
    ax=Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    p=np.linspace(0,1)#默认生成50
    q=np.linspace(0,1)
    P,Q=np.meshgrid(p,q)
    R=np.hstack((P.ravel().reshape(2500,1),Q.ravel().reshape(2500,1)))#ravel 扁平化，变成一维数组
    Z=net.inference(R)#hstack 在水平方向上平铺 R变成2500*2的数组,Z应该是2500*1
    Z=Z.reshape(50,50)
    ax.plot_surface(P,Q,Z,cmap='rainbow')
    plt.show()
def draw_2D(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()
if __name__=='__main__':
    file_name='G:\Git\CNNlearn\B-教学案例与实践\B6-神经网络基本原理简明教程\Data\ch05.npz'
    #读取数据
    reader=DataReader_1_0(file_name)
    reader.ReadData()
    reader.NormalizeX()#归一化
    reader.NormalizeY()#归一化Y
    #创建神经元
    params=HyperParameters_1_0(2,1,eta=0.01,max_epoch=50,batch_size=10,eps=1e-5)
    net=NeuralNet(params)
    #开始训练
    net.Train(reader)
    #预测一下
    x=np.array([15,93]).reshape(1,2)
    #print(net.inference(x))#对归一化之后的X没做任何处理
    #W,B=DeNormalizeWeightsBias(net,reader)
    #print(W,B)
    #Z=np.dot(x,W)+B#通过还原真实WB来预测
    #print(Z)
    x=reader.NormalizePredicateData(x)#通过将要预测的x归一化，进行预测
    #print(net.inference(x))
    z=net.inference(x)
    z=DeNormalizeY(z,reader)
    print(z)
    draw_3D(net,reader)
    #net.draw_3D(reader)
    '''
    a=np.array([[1],[2],[3],[4],[1],[2],[3],[4],[5]]).reshape(3,3)
    print(a)
    a=[1,2,3]
    b=[-1,-2,-3]
    c=np.meshgrid(a,b)
    print(c)
    '''
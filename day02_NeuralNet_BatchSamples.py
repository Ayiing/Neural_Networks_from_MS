from LoadData import *
import numpy as np
class NeuralNet(object):
    def __init__(self,params):
        self.params=params
        self.w=0#该问题样本只有一个特征，只有一个w
        self.b=0#该问题只有一个神经元，只有一个b
    def __forwardBatch(self,batch_x):
        #Z=X*self.w+self.b
        batch_z=np.dot(batch_x,self.w)+self.b#np.dot:矩阵乘法
        return batch_z
    def __backwardBatch(self,batch_x,batch_y,batch_z):
        dZ=batch_z-batch_y
        dw=np.dot(batch_x.T,dZ)/self.params.batch_size
        db=dZ.sum(axis=0)/self.params.batch_size
        return dw,db
    def __update(self,dw,db):
        self.w=self.w-self.params.eta*dw
        self.b=self.b-self.params.eta*db
    def Train(self,dataReader):
        if self.params.batch_size<0:
            self.params.batch_size=dataReader.num_train
        m=int(dataReader.num_train/self.params.batch_size)#一次计算batch_size个样本为1批，m批后会有剩的个体，不够batch_size个，不予理会，每个epoch，重排Xtrain
        for epoch in range(self.params.max_epoch):
            dataReader.Shuffle()#每个epoch之前，数据重排一下，这样每次最后剩的那几个数据就不一样了
            for i in range(m):
                batch_x,batch_y=dataReader.GetBatchTrainSamples(self.params.batch_size,i)
                batch_z=self.__forwardBatch(batch_x)
                dw,db=self.__backwardBatch(batch_x,batch_y,batch_z)
                self.__update(dw,db)
                dZ=batch_z-batch_y
                loss=(dZ**2).sum(axis=0)*1.0/m/2
                print("loss:",loss)
                if loss<self.params.eps:
                    break
            print("epoch%d:\n w:%f,b:%f"%(epoch,self.w,self.b))
            if loss<self.params.eps:
                break
        print("result:\n w:%f,b:%f"%(self.w,self.b))
    def inference(self,x):
        return self.__forwardBatch(x)
if __name__=='__main__':
    params=HyperParameters_1_0(1,1,eta=0.3,max_epoch=100,batch_size=-1,eps=0.02)
    #构建一个神经元
    net=NeuralNet(params)
    #读取数据
    file_name='f:/pyzo/network/ch04.npz'
    reader=DataReader_1_0(file_name)
    reader.ReadData()
    #开始训练
    net.Train(reader)
    #print(net.w,net.b)
    #看一下结果
    draw(net,reader)
'''
a=np.array([ [1],[2],[5] ])
print(a.shape[0],a.shape[1])#shape[0] 行数，shape[1] 列数
b=3
print(np.dot(a,b))
a=np.array([ [1,2],[3,3],[1,4] ])
print(a.sum(axis=0))#压缩行
print(a.sum(axis=1))#压缩列 然后变成了行
print(a.sum(axis=1,keepdims=True).sum(axis=0))#keepdims 保存列的维度
print(np.dot(3,5))#=15
'''


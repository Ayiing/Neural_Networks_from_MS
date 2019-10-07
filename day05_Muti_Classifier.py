from NeuralNet import *
def inference(net,reader,xt_raw):
    xt=reader.NormalizePredicateData(xt_raw)
    n=xt.shape[0]
    #把预测点在途中标出来
    for i in range(n):
        plt.scatter(xt[i,0],xt[i,1],marker='*',c='y')
    output=net.inference(xt)
    result=np.argmax(output,axis=1)+1
    print(output)
    print(result)
def draw_result(net,reader):
    #先把所有散点画出来
    '''
    X=reader.XTrain#明白哪里错了，Train在每一次epoch都会shuffle一遍，XTrain和Yraw已经不对应了
    Y=reader.YRaw
    for i in range(reader.num_train):
        #print(int(Y[i,0]))#知道错在哪了，Y[i,0]里面存的是float
        num=int(Y[i,0])
        if num == 1:
            plt.scatter(X[i,0],X[i,1],marker='x',c='g')
        elif num == 2:
            plt.scatter(X[i,0],X[i,1],marker='o',c='r')
        else:
            plt.scatter(X[i,0],X[i,1],marker='v',c='k')
    '''
    X,Y=reader.GetWholeTrainSamples()
    for i in range(reader.num_train):
        category=np.argmax(Y[i])
        if category==0:
            plt.scatter(X[i,0],X[i,1],marker='x',c='g')
        elif category==1:
            plt.scatter(X[i,0],X[i,1],marker='o',c='r')
        else:
            plt.scatter(X[i,0],X[i,1],marker='v',c='k')
    #画分界线
    b12=(net.B[0,1]-net.B[0,0])/(net.W[1,0]-net.W[1,1])
    w12=(net.W[0,1]-net.W[0,0])/(net.W[1,0]-net.W[1,1])
    b13=(net.B[0,2]-net.B[0,0])/(net.W[1,0]-net.W[1,2])
    w13=(net.W[0,2]-net.W[0,0])/(net.W[1,0]-net.W[1,2])
    b23=(net.B[0,2]-net.B[0,1])/(net.W[1,1]-net.W[1,2])
    w23=(net.W[0,2]-net.W[0,1])/(net.W[1,1]-net.W[1,2])
    #
    x=np.linspace(0,1,2)
    y=w13*x+b13
    p13,=plt.plot(x,y,c='g')
    #
    x=np.linspace(0,1,2)
    y=w23*x+b23
    p23,=plt.plot(x,y,c='r')
    #
    x=np.linspace(0,1,2)
    y=w12*x+b12
    p12,=plt.plot(x,y,c='b')
    #
    plt.legend([p13,p23,p12],["13","23","12"])#显示图例
    plt.axis([-0.1,1.1,-0.1,1.1])#设定x轴[-0.1,1.1],y轴同样
    #plt.show()
if __name__=='__main__':
    file_name='G:\Git\CNNlearn\B-教学案例与实践\B6-神经网络基本原理简明教程\Data\ch07.npz'
    #读取数据
    num_category=3#独热编码需要三个位
    reader=DataReader_1_0(file_name)
    reader.ReadData()
    reader.NormalizeX()#归一化
    reader.ToOneHot(num_category,base=1)#标签值处理
    #创建神经元
    params=HyperParameters_1_0(eta=0.1,max_epoch=100,batch_size=10,eps=1e-3,net_type=NetType.MultipleClassifier)
    input=2
    output=3
    net=NeuralNet(params,input,output)
    #开始训练
    net.Train(reader,checkpoint=1)
    draw_result(net,reader)
    x=np.array([5,1,7,6,5,6,2,7]).reshape(4,2)
    inference(net,reader,x)
    plt.show()
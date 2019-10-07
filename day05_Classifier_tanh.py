from NeuralNet import *
if __name__=='__main__':
    file_name='G:\Git\CNNlearn\B-教学案例与实践\B6-神经网络基本原理简明教程\Data\ch06.npz'#'F:\pyzo\network\data\ch06.npz'
    #读取数据
    reader=DataReader_1_0(file_name)
    reader.ReadData()
    print(reader.YTrain)
    reader.ToZeroOne()
    print(reader.YTrain)
    #归一化？数据做完处理了
    #创建神经元
    params=HyperParameters_1_0(eta=0.1,max_epoch=1000,batch_size=10,eps=1e-3,net_type=NetType.BinaryClassifier_tanh)
    input=2
    output=1
    net=NeuralNet(params,input,output)
    #开始训练
    net.Train(reader,checkpoint=10)
    x=np.array([0.58,0.92,0.62,0.55,0.39,0.29]).reshape(3,2)
    a=net.inference(x)
    print(a)
    draw(net,reader)#里面有show
    draw_predicate(x)#没有show
    plt.show()
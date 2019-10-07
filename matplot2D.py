import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def draw_3D():
    fig=plt.figure()#空白窗口
    ax=Axes3D(fig)#画个3D
    u=np.linspace(-1,1,100)#均匀取点
    v=np.linspace(-1,1,100)
    X,Y=np.meshgrid(u,v)#生成网格点坐标矩阵
    R=np.zeros([len(u),len(v)])# 初始化一个多维数组，注意参数格式
    for i in range(len(u)):
        for j in range(len(v)):
            R[i,j]=np.sin(X[i,j]*Y[i,j])#X[i,j]**2+np.sin(Y[i,j])**2
    ax.plot_surface(X,Y,R,cmap='rainbow')
    plt.show()
def draw_2D():
    x=np.linspace(-10,10,100)
    y=[]
    for i in range(len(x)):
        y.append( 4.00*(1-np.sin(x[i])) )
    plt.plot(x,y)
    plt.show()
'''
X=[]
for i in range(-25,50):
    X.append(i)
Y=[]
for i in X:
   Y.append(i*i*i)
plt.plot(X,Y)
x1=[1,13,45,4]
y1=[]
for i in x1:
    y1.append(i*i*i)
plt.plot(x1,y1)
plt.show()
'''
draw_3D()
draw_2D()
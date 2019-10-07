import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def func(w,b,z):
    #n是迭代次数
    error=1e-6
    i=1
    deltZ=1
    while(abs(deltZ)>error):
        x=2*w+3*b
        y=2*b+1
        z1=x*y
        print(i,"w:",w,"b:",b,z1)
        #如果w，b对z的变化贡献各占一半的话
        deltZ=(z-z1)
        zw=2*y
        zb=y*3+x*2
        #'''
        deltW=deltZ/2/zw
        deltB=deltZ/2/zb
        #'''
        '''
        deltW=deltZ/zw/zw
        deltB=deltZ/zb/zb #至少能跑出来
        '''
        '''
        deltW=deltZ*zb/(zw+zb)
        deltB=deltZ*zw/(zw+zb)#很离谱,根本跑不出来
        '''
        '''
        deltW=deltZ* ( zw/(zw+zb) )/zw
        deltB=deltZ* ( zb/(zw+zb) )/zb
        '''
        w=w+deltW
        b=b+deltB
        i+=1
#func(3,4,150)
def playgame(x,t):
    error=1e-6
    i=1
    delt_c=1
    while(abs(delt_c)>error):
        a=x*x
        b=math.log(a)
        c=math.sqrt(b)
        cx=1/(x*math.sqrt(math.log(x*x)))
        delt_c=t-c
        delt_x=delt_c/cx
        x+=delt_x
        print(i,"x:",x,"c:",c)
        i+=1
#playgame(2,2.13)
def gradient(x0,y0):
    #y=x*x+sin^2(y)
    #(2*x,2*siny*cosy)
    return np.array([2*x0,2*math.sin(y0)*math.cos(y0)])
def draw_3D(x,y,z):
    fig=plt.figure()#空白窗口
    ax=Axes3D(fig)#画个3D
    u=np.linspace(-3,3,100)#均匀取点
    v=np.linspace(-3,3,100)
    X,Y=np.meshgrid(u,v)#生成网格点坐标矩阵
    R=np.zeros([len(u),len(v)])# 初始化一个多维数组，注意参数格式
    for i in range(len(u)):
        for j in range(len(v)):
            R[i,j]=X[i,j]**2+math.sin(Y[i,j])**2
    ax.plot_surface(X,Y,R,cmap='rainbow')
    plt.plot(x,y,z,c='black')
    plt.show()
if __name__=='__main__':
    x=3
    y=1
    j=pow(x,2)+pow(math.sin(y),2)
    eta=0.99
    error=1e-2
    X=[]
    Y=[]
    J=[]
    while j>error:
        X.append(x)
        Y.append(y)
        J.append(j)
        s0=np.array([x,y])
        j=x**2+math.sin(y)**2
        gra=gradient(x,y)
        #s1=np.subtract(s0,eta*gra)# 数 * array = 数 * array的每一个元素
        s1=s0-eta*gra
        x,y=s1[0],s1[1]
        print(s1,j)
    draw_3D(X,Y,J)
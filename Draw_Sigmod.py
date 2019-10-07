import matplotlib.pyplot as plt
import numpy as np
def draw_Sigmod():
    x=np.linspace(-5,5)
    y=1/(1+np.exp(-x))
    '''
    y=[]
    for i in x:
        z=1/(1+np.exp(-i))
        y.append(z)
    '''
    y_=y*(1-y)
    plt.plot(x,y)
    plt.plot(x,y_)
    plt.show()
draw_Sigmod()
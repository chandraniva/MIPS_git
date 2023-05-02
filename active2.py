import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

startTime = datetime.now()

@jit
def nbr2d(k,L):    
    nbr=np.zeros((L*L,4)) 
    for i in range(L):
            for j in range(L):
                k =  j*L +i

                nbr[k,0]=  ((j+1)%L)*L+i    
                nbr[k,1]=  j*L + ((i+1)%L)   
                nbr[k,2]=  i + L*((j-1+L)%L)
                nbr[k,3]=  ((i-1+L)%L) +j*L 
    return nbr 


@jit
def ordr_param(s):
    xp = len(np.where(s==1)[0])
    xm = len(np.where(s==3)[0])
    yp = len(np.where(s==2)[0])
    ym = len(np.where(s==4)[0])
    return np.sqrt((xp-xm)**2 + (yp-ym)**2)/(xp+xm+yp+ym)
        


@jit
def dot(x,y):
    x_ang = (x-1)*np.pi/2
    if y>0:
        y_ang = (y-1)*np.pi/2
        return np.cos(x_ang-y_ang)
    else:
        return 0

@jit
def H(s,i):
    eng = 0
    for k in range(4):
        eng += lm * dot(s[nbr[i,k]],s[i])
        eng += int(s[nbr[i,k]]>0) 
    return -eng
    
@jit
def nbr_vec(s,i):
    Dx, Dy = 0, 0
    for k in range(4):
        sj = s[nbr[i,k]]
        if sj>0:
            Dx += np.cos((sj-1)*np.pi/2)
            Dy += np.sin((sj-1)*np.pi/2)
    return Dx, Dy

@jit
def find_dir(x,y):
    if np.abs(x)>np.abs(y):
        if x>0:
            return 1
        if x<0:
            return 3
    elif np.abs(y)>np.abs(x):
        if y>0:
            return 2
        if y<0:
            return 4
    else:
        if x>0 and y>0:
            if np.random.rand()<0.5:
                return 1
            else:
                return 2
        elif x>0 and y<0:
            if np.random.rand()<0.5:
                return 1
            else:
                return 4
        elif x<0 and y>0:
            if np.random.rand()<0.5:
                return 3
            else:
                return 2
        else:
            if np.random.rand()<0.5:
                return 3
            else:
                return 4
        

@jit
def update(s):
    for r in range(sqL):
        i= int(np.random.rand()*sqL)
        if s[i]>0:
            if np.random.rand()<0.5: #spin change
                Dx, Dy = nbr_vec(s,i)
                if np.abs(Dx) + np.abs(Dy) > 0:
                    D = find_dir(Dx,Dy)
                    eng_i = H(s,i)
                    temp = s[i]
                    s[i] = D
                    eng_f = H(s,i)
                    dE = eng_f - eng_i
                    s[i] = temp
                    if dE>0:
                        if np.random.rand()<np.exp(-beta*dE):
                            s[i] = D
                    else:
                        s[i] = D

                
            else: #movement
                j = nbr[i,s[i]-1]
                if s[j] == 0:    
                    eng_i = H(s,i)
                    s[j] = s[i]
                    s[i] = 0
                    eng_f = H(s,j)
                    dE = eng_f - eng_i
                    s[i] = s[j]
                    s[j] = 0
                    if dE>0:
                        if np.random.rand()<np.exp(-beta*dE):
                            s[j] = s[i]
                            s[i] = 0
                    else:
                        s[j] = s[i]
                        s[i] = 0
                        
    return s


def viz(s):
    s = s.reshape(L,L)
    p1 = np.where(s == 1)
    p2 = np.where(s == 2)
    p3 = np.where(s == 3)
    p4 = np.where(s == 4)
    
    fig = plt.figure()
    fig = plt.figure(figsize = (25,25))
    ax1 = fig.add_subplot(1,1,1, adjustable='box', aspect=1.0)
    sc = L
    ax1.quiver(p1[0],p1[1],np.ones(len(p1[0])),np.zeros(len(p1[1])),scale=sc)
    ax1.quiver(p2[0],p2[1],np.zeros(len(p2[0])),np.ones(len(p2[1])),scale=sc)
    ax1.quiver(p3[0],p3[1],-np.ones(len(p3[0])),np.zeros(len(p3[1])),scale=sc)
    ax1.quiver(p4[0],p4[1],np.zeros(len(p4[0])),-np.ones(len(p4[1])),scale=sc)
    
    ax1.set_xlim(0,L)
    ax1.set_ylim(0,L)
    plt.show()
    

L = 32
rho = 0.3
N = rho*L*L
sqL = L*L   
beta = 1
lm = 1
time = 1000
op = np.zeros(time)

for k in range(sqL):
    nbr=nbr2d(k,L)
nbr = nbr.astype('int16')
print("done...")

init = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                        p=[1-rho, rho/4, rho/4, rho/4, rho/4])


for t in range(time):
    if t%1 == 0:
        viz(init)
    op[t] = ordr_param(init)
    init = update(init)
    
viz(init)

fig2 = plt.figure()
fig2 = plt.figure(figsize = (25,15))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(np.arange(time),op,'.-')
ax2.set_ylim(0,1)
ax2.set_xscale('log')
plt.show()




print("Execution time:",datetime.now() - startTime)
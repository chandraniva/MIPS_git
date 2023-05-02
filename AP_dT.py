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
                
                nbr[k,0]=   j*L + ((i+1)%L)    
                nbr[k,1]=  i + L*((j+1)%L)       
                nbr[k,2]= ((i-1+L)%L) +j*L    
                nbr[k,3]= ((j-1+L)%L)*L+i    
    return nbr 


@jit
def ordr_param(s):
    xp = len(np.where(s==1)[0])
    xm = len(np.where(s==3)[0])
    yp = len(np.where(s==2)[0])
    ym = len(np.where(s==4)[0])
    return np.sqrt((xp-xm)**2 + (yp-ym)**2)/(xp+xm+yp+ym)
        


@jit
def kronecker(x,y):
    if x == y:
        return 1
    else:
        return 0


@jit
def H(s,i):
    eng = 0
    for k in range(4):
        eng += J * kronecker(s[nbr[i,k]],s[i])
        eng += int(s[nbr[i,k]]>0) 
    return -eng
    
 


@jit
def update(s,beta):
    for r in range(sqL):
        i= int(np.random.rand()*sqL)
        if s[i]>0:
            if np.random.rand()<0.5: #spin change
                    eng_i = H(s,i)
                    temp = s[i]
                    D = np.random.randint(1,5)
                    while(D==s[i]):
                        D = np.random.randint(1,5)
                    s[i] = D
                    eng_f = H(s,i)
                    dE = eng_f - eng_i
                    if dE>0:
                        if np.random.rand()<1-np.exp(-beta*dE):
                            s[i] = temp
                
            else: #movement
                j = nbr[i,s[i]-1]
                if s[j] == 0:    
                    eng_i = H(s,i)
                    s[j] = s[i]
                    s[i] = 0
                    eng_f = H(s,j)
                    dE = eng_f - eng_i
                    if dE>0:
                        if np.random.rand()<1-np.exp(-beta*dE):
                            s[i] = s[j]
                            s[j] = 0
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
    ax1.quiver(p1[1],p1[0],np.ones(len(p1[0])),np.zeros(len(p1[1])),scale=sc)
    ax1.quiver(p2[1],p2[0],np.zeros(len(p2[0])),np.ones(len(p2[1])),scale=sc)
    ax1.quiver(p3[1],p3[0],-np.ones(len(p3[0])),np.zeros(len(p3[1])),scale=sc)
    ax1.quiver(p4[1],p4[0],np.zeros(len(p4[0])),-np.ones(len(p4[1])),scale=sc)
    
    ax1.set_xlim(0,L)
    ax1.set_ylim(0,L)
    plt.show()
    

L = 16
rho = 0.6
N = rho*L*L
sqL = L*L   
beta = 1.3
beta2 = 1.1
J = 1
time = 10000
ens = 5000
trlx = 100000

op = np.zeros(time)

for k in range(sqL):
    nbr=nbr2d(k,L)
nbr = nbr.astype('int16')


init = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                p=[1-rho, rho, 0, 0, 0])

for t in range(trlx):
    init = update(init,beta)
    
def ss(conf,beta):
    for i in range(10):
        conf = update(conf,beta)
    return conf
    

for e in range(ens):    
    init = ss(init,beta)
    s = np.copy(init)
    for t in range(time):
        op[t] += ordr_param(s)
        s = update(s,beta2)



fig2 = plt.figure()
fig2 = plt.figure(figsize = (8,6))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(np.arange(time),op/ens,'.-',label='beta='+str(beta2))
ax2.set_ylim(0.1,1)
ax2.legend()
plt.show()


np.savetxt("AP_dT"+str(trlx)+"L="+str(L)+"_beta1="+str(beta)+"_beta2="+str(beta2)+
           "_ens="+str(ens)+"_rho="+str(rho)+"_J="+str(J)+".txt",op)


print("Execution time:",datetime.now() - startTime)

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
    

L = 8
rho = 0.6
N = rho*L*L
sqL = L*L   
J = 1


for k in range(sqL):
    nbr=nbr2d(k,L)
nbr = nbr.astype('int16')
print("done...")

trlx = 100000
ens = 1000000
beta = np.arange(1.0,1.2,0.025)
op1 = np.zeros_like(beta)
op2 = np.zeros_like(beta)
op4 = np.zeros_like(beta)
    
i = 0
for b in beta:

    init = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                        p=[1-rho, rho/4, rho/4, rho/4, rho/4])
    
    for t in range(trlx):
        init = update(init,b)
        
    for e in range(ens):
        op = ordr_param(init)
        op1[i] += op
        op2[i] += op*op
        op4[i] += op*op*op*op
        init = update(init,b)
        
    op1[i] = op1[i]/ens
    op2[i] = op2[i]/ens
    op4[i] = op4[i]/ens
    
    print(b,op1[i],op2[i],op4[i])
    
    i += 1
    

print(beta)
print(op1)
print(op2)
print(op4)

print("Execution time:",datetime.now() - startTime)

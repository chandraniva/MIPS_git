import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit
# import numba as nb
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
def init_arr(sqL,rho):
    init = np.empty(sqL,dtype=int)
    for i in range(sqL):
        if np.random.rand()<rho:
            init[i] = 1
        else:
            init[i] = 0
    return init

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




L = 64
rho = 0.6
N = rho*L*L
sqL = L*L   
J = 1


for k in range(sqL):
    nbr=nbr2d(k,L)
nbr = nbr.astype('int16')
print("done...")


time = 2000
ens = 100
beta_c = 1.30
beta_u = beta_c + 0.05
beta_d = beta_c - 0.05


op = np.zeros(time)
op_u = np.zeros(time)
op_d = np.zeros(time)

for e in range(ens):
    init = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                        p=[1-rho, rho, 0, 0, 0])
    init_u = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                        p=[1-rho, rho, 0, 0, 0])
    init_d = np.random.choice([0, 1, 2, 3, 4], size=(sqL,),\
                        p=[1-rho, rho, 0, 0, 0])
    
    for t in range(time):
        op[t] += ordr_param(init)
        init = update(init,beta_c)
        
        op_u[t] += ordr_param(init_u)
        init_u = update(init_u,beta_u)
        
        op_d[t] += ordr_param(init_d)
        init_d = update(init_d,beta_d)
    
op = op/ens
op_u = op_u/ens
op_d = op_d/ens

fig2 = plt.figure()
fig2 = plt.figure(figsize = (10,8))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(np.arange(time),op,'.-',label='beta='+str(beta_c))
ax2.plot(np.arange(time),op_u,'.-',label='beta='+str(beta_u))
ax2.plot(np.arange(time),op_d,'.-',label='beta='+str(beta_d))
ax2.set_ylim(0.01,1)
ax2.legend()
ax2.loglog()
plt.savefig("OPvstime.png",dpi=500)
plt.show()




print("Execution time:",datetime.now() - startTime)

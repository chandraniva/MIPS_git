import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
from tqdm import tqdm

startTime = datetime.now()

@jit
def nbr2d(k,Lx,Ly):    
    nbr=np.zeros((Lx*Ly,4)) 
    for i in range(Lx):
            for j in range(Ly):
                k =  j*Lx +i
                nbr[k][0]=  j*Lx+ ((i+1)%Lx)
                nbr[k][1]=  ((j+1)%Ly)*Lx+i
                nbr[k][2]=  ( (i-1+Lx)%Lx) +j*Lx
                nbr[k][3]=  i + Lx*((j-1+Ly)%Ly)
    return nbr 



def init_dis(N,sqL):
    k=0
    s = np.zeros(sqL,dtype=int)
    while(k<N):
        i = int(np.random.rand()*sqL)
        if s[i] == 0:
            s[i] = np.random.randint(1,5)
            k+=1
    return s

def init_ord(N,sqL):
    s = np.zeros(sqL,dtype=int)
    for i in range(int(sqL/4), int(3/4*sqL)):
        s[i] = np.random.randint(1,5)
    return s


def init_ising(N,sqL):
    k=0
    s = np.zeros(sqL,dtype=int)
    while(k<N):
        i = int(np.random.rand()*sqL)
        if s[i] == 0:
            s[i] = 1
            k+=1
    return s
    

@jit
def ordr_param(s):
    mg = 0
    for j in range(Ly):
        n = 0
        for i in range(Lx):
            n+= int(s[j*Lx + i] > 0)
        mg+=np.abs(n/Lx-0.5)
    return 2*mg/Ly

@jit
def H(s,i):
    eng = 0
    for k in range(4):
        eng += int(s[nbr[i,k]]>0) 
    return -eng
 

@jit
def update_ising(s,beta):
    for r in range(sqL):
        i= int(np.random.rand()*sqL)
        if s[i]>0:
            j = nbr[i,np.random.randint(1,5)]
            if s[j] == 0:    
                eng_i = H(s,i)
                s[j] = s[i]
                s[i] = 0
                eng_f = H(s,j)
                if eng_f > eng_i:
                    if np.random.rand()<1-np.exp(-beta*(eng_f-eng_i)):
                        s[i] = s[j]
                        s[j] = 0
    return s


@jit
def update_q(s,beta,q):

    if q>1:
        dt = 1/q
        for r in range(sqL):
            i= int(np.random.rand()*sqL)
            if s[i]>0:
                if np.random.rand()<0.5:
                    s[i] = (np.random.randint(0,3) + s[i])%4 + 1
                else: #movement
                    j = nbr[i,s[i]-1]
                    if s[j] == 0:    
                        eng_i = H(s,i)
                        s[j] = s[i]
                        s[i] = 0
                        eng_f = H(s,j)
                        if eng_f > eng_i:
                            if np.random.rand()<1-np.exp(-beta*(eng_f-eng_i))*dt:
                                s[i] = s[j]
                                s[j] = 0
                        elif np.random.rand()<1-dt:
                                s[i] = s[j]
                                s[j] = 0     
                                
    elif q>0:
        for r in range(sqL):
            i= int(np.random.rand()*sqL)
            if s[i]>0:
                if np.random.rand()<0.5:
                    if np.random.rand()<q: #spin change
                            s[i] = (np.random.randint(0,3) + s[i])%4 + 1
                else: #movement
                        j = nbr[i,s[i]-1]
                        if s[j] == 0:    
                            eng_i = H(s,i)
                            s[j] = s[i]
                            s[i] = 0
                            eng_f = H(s,j)
                            if eng_f > eng_i:
                                if np.random.rand()<1-np.exp(-beta*(eng_f-eng_i)):
                                    s[i] = s[j]
                                    s[j] = 0
    else:
        for r in range(sqL):
            i= int(np.random.rand()*sqL)
            if s[i]>0:
                j = nbr[i,np.random.randint(1,5)]
                if s[j] == 0:    
                    eng_i = H(s,i)
                    s[j] = s[i]
                    s[i] = 0
                    eng_f = H(s,j)
                    if eng_f > eng_i:
                        if np.random.rand()<1-np.exp(-beta*(eng_f-eng_i)):
                            s[i] = s[j]
                            s[j] = 0
            
    return s
                


Lx, Ly = 32, 64
rho = 1/2
N = int(rho*Lx*Ly)
sqL = Lx*Ly
q=0.005
beta = 0

for k in range(sqL):
    nbr=nbr2d(k,Lx,Ly)
nbr = nbr.astype('int16')
print("done...")


print('q='+str(q)+',lx='+str(Lx)+',ly='+str(Ly))

time = 100000 #2*int(1e4)    
ens = 10
op_ord = np.zeros(time)
op_dis = np.zeros(time)

for e in tqdm(range(ens)):
    s1 = init_dis(N,sqL)
    s2 = init_ord(N,sqL)
    for t in range(time):
        op_dis[t] += ordr_param(s1)
        op_ord[t] += ordr_param(s2)
        s1 = update_q(s1,beta,q)
        s2 = update_q(s2,beta,q)
        
    np.save('q_time_'+str(time)+'_q'+str(q)+'_lx='+str(Lx)+'_ly='+str(Ly)+
            '.npy',np.vstack((op_ord/(e+1),op_dis/(e+1))))   
        
op_dis = op_dis/ens
op_ord = op_ord/ens

np.save('q_time_'+str(time)+'_q'+str(q)+'_lx='+str(Lx)+'_ly='+str(Ly)+
        '.npy',np.vstack((op_ord,op_dis)))    

plt.plot(op_ord)
plt.plot(op_dis)
plt.ylim(0,1)
plt.show()   
    
    
print("Execution time:",datetime.now() - startTime)
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime

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
    return mg/Ly


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
    


Lx, Ly = 16, 32
rho = 1/2
N = int(rho*Lx*Ly)
sqL = Lx*Ly
q = -1

for k in range(sqL):
    nbr=nbr2d(k,Lx,Ly)
nbr = nbr.astype('int16')
print("done...")

trlx = 1000000
ens = 1000000
temp = np.arange(0.4,0.91,0.1)
op1 = np.zeros_like(temp)
op2 = np.zeros_like(temp)
op4 = np.zeros_like(temp)

print('q='+str(q)+',lx='+str(Lx)+',ly='+str(Ly))

i = 0
for tm in temp:
    
    b = 1/tm
    # s = init_ising(N,sqL)
    s = init_dis(N,sqL)
    
    for t in range(trlx):
        # s = update_ising(s,b)
        s = update_q(s,b,q)
        
    for e in range(ens):
        op = ordr_param(s)
        op1[i] += op
        op2[i] += op*op
        op4[i] += op*op*op*op
        s = update_q(s,b,q)
        # s = update_ising(s,b)
        
    op1[i] = op1[i]/ens
    op2[i] = op2[i]/ens
    op4[i] = op4[i]/ens
    
    print(tm,op1[i],op2[i],op4[i])
    
    i += 1
    
    
np.save('q='+str(q)+'_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+
        '.npy',np.vstack(temp,op1,op2,op4))    
    
    
print("Execution time:",datetime.now() - startTime)
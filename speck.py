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
    
@jit
def update_speck(s,wp1,w1):
    for r in range(sqL):
        i= int(np.random.rand()*sqL)
        if s[i]>0:
            if np.random.rand()<0.5: #movement
                if np.random.rand() < wp1:
                    dirn = s[i]
                # else:
                #     dirn = (np.random.randint(0,3) + s[i])%4 + 1
                    
                j = nbr[i,dirn-1]
                if s[j]==0:
                    s[j] = s[i]
                    s[i] = 0
                
            elif np.random.rand()<w1: #flip/tumble
                s[i] = (2*np.random.randint(0,2) -2 + s[i])%4 + 1
                
    return s
                

def viz(s,t):
    s = s.reshape(Ly,Lx)
    p1 = np.where(s == 1)
    p2 = np.where(s == 2)
    p3 = np.where(s == 3)
    p4 = np.where(s == 4)
    
    fig = plt.figure()
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(1,1,1, aspect=1)
    ax1.quiver(p1[1],p1[0],np.ones(len(p1[0])),np.zeros(len(p1[1])),scale = Ly,color='g')
    ax1.quiver(p2[1],p2[0],np.zeros(len(p2[0])),np.ones(len(p2[1])),scale = Ly,color='r')
    ax1.quiver(p3[1],p3[0],-np.ones(len(p3[0])),np.zeros(len(p3[1])),scale = Ly,color='magenta')
    ax1.quiver(p4[1],p4[0],np.zeros(len(p4[0])),-np.ones(len(p4[1])),scale=Ly,color='b')
    
    ax1.set_xlim(0,Lx)
    ax1.set_ylim(0,Ly)
    ax1.grid(axis='both',linestyle='-',color='slategrey')
    ax1.set_xticks(np.arange(0,Lx))
    ax1.set_yticks(np.arange(0,Ly))
    plt.title('time='+str(t))
    plt.show()
    

"==========================   visualization  ============================"

    
# Lx, Ly = 16, 32
# rho = 1/2
# N = int(rho*Lx*Ly)
# sqL = Lx*Ly
# s = init_ord(N,sqL)
# wp1 = 5
# wp = wp1/(3+wp1)
# w1 = 0.1/(3+wp1)


# for k in range(sqL):
#     nbr=nbr2d(k,Lx,Ly)
# nbr = nbr.astype('int16')


# time = 10000001

# for t in range(time):
#     if t == 10 or t == 100 or t == 1000 or t == 10000 or t == 100000 or t == int(1e6) or t == int(1e7):
#         viz(s,t)
    
#     update_speck(s,wp,w1)






"==========================  with control params  ============================"

# Lx, Ly = 16, 32
# rho = 1/2
# N = int(rho*Lx*Ly)
# sqL = Lx*Ly

# for k in range(sqL):
#     nbr=nbr2d(k,Lx,Ly)
# nbr = nbr.astype('int16')

# trlx = 1000000
# ens = 10000000
# wp = np.arange(4.5,5,0.1)


# op1 = np.zeros_like(wp)
# op2 = np.zeros_like(wp)
# op4 = np.zeros_like(wp)


# i = 0
# for tm in wp:
    
#     wp1 = tm/(3+tm)
#     w1 = 0.1/(3+tm)
    
#     s = init_dis(N,sqL)
    
#     for t in range(trlx):
#         s = update_speck(s,wp1,w1)
        
#     for e in range(ens):
#         op = ordr_param(s)
#         op1[i] += op
#         op2[i] += op*op
#         op4[i] += op*op*op*op
#         s = update_speck(s,wp1,w1)
        
#     op1[i] = op1[i]/ens
#     op2[i] = op2[i]/ens
#     op4[i] = op4[i]/ens
    
#     print(tm,op1[i],op2[i],op4[i])
    
#     i += 1
    
# np.save('sp_wp1'+str(wp1)+'_lx='+str(Lx)+'_ly='+str(Ly)+'_w1='+str(w1)+
#         '.npy',np.vstack(temp,op1,op2,op4))    
    
    
"===========================  with time  ===================================="
    
Lx, Ly = 32, 64
rho = 1/2
N = int(rho*Lx*Ly)
sqL = Lx*Ly
wp1 = 3
wp = wp1/(3+wp1)
w1 = 0.1/(3+wp1)
    
for k in range(sqL):
    nbr=nbr2d(k,Lx,Ly)
nbr = nbr.astype('int16')

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
        s1 = update_speck(s1,wp,w1)
        s2 = update_speck(s2,wp,w1)
        
    np.save('sp_time_'+str(time)+'_wp1'+str(wp1)+'_lx='+str(Lx)+'_ly='+str(Ly)+
            '_w1='+str(w1)+'.npy',np.vstack((op_ord/(e+1),op_dis/(e+1))))   
        
op_dis = op_dis/ens
op_ord = op_ord/ens
    
np.save('sp_time_'+str(time)+'_wp1'+str(wp1)+'_lx='+str(Lx)+'_ly='+str(Ly)+
        '_w1='+str(w1)+'.npy',np.vstack((op_ord,op_dis)))    

plt.plot(op_ord)
plt.plot(op_dis)
plt.ylim(0,1)
plt.show()    
    
print("Execution time:",datetime.now() - startTime)
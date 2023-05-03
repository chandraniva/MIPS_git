import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
import matplotlib.animation as animation

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
    ax1.quiver(p1[1],p1[0],np.ones(len(p1[0])),np.zeros(len(p1[1])),scale = Lx)
    ax1.quiver(p2[1],p2[0],np.zeros(len(p2[0])),np.ones(len(p2[1])),scale = Lx)
    ax1.quiver(p3[1],p3[0],-np.ones(len(p3[0])),np.zeros(len(p3[1])),scale = Lx)
    ax1.quiver(p4[1],p4[0],np.zeros(len(p4[0])),-np.ones(len(p4[1])),scale=Lx)
    
    ax1.set_xlim(0,Lx)
    ax1.set_ylim(0,Ly)
    plt.title('time='+str(t))
    plt.show()
    
    
def viz_ising(s,t):
    s[np.where(s>0)]=1
    s = s.reshape(Ly,Lx)
    fig = plt.figure()
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(1,1,1, aspect=1)
    ax1.imshow(s,cmap='binary')
    
    # ax1.set_xlim(0,Lx)
    # ax1.set_ylim(0,Ly)
    plt.title('time='+str(t))
    plt.show()
    
# @jit
def pos_vel(s,Lx,Ly):
    s = s.reshape(Ly,Lx)
    xs, ys = np.where(s>0)[0], np.where(s>0)[1]
    vx = np.zeros_like(xs)
    vy = np.zeros_like(xs)
    for j in range(len(xs)):
        spin = s[xs[j],ys[j]]
        if spin == 1:
            vx[j] = 1
            vy[j] = 0
        if spin == 2:
            vx[j] = 0
            vy[j] = 1
        if spin == 3:
            vx[j] = -1
            vy[j] = 0
        if spin == 4:
            vx[j] = 0
            vy[j] = -1   
    return xs,ys,vx,vy
    

Lx, Ly = 32, 64
rho = 1/2
N = int(rho*Lx*Ly)
sqL = Lx*Ly
temp = 0.5
beta = 1/temp
time = int(1e6) + 1
q = 0.05

for k in range(sqL):
    nbr=nbr2d(k,Lx,Ly)
nbr = nbr.astype('int16')

s = init_ord(N,sqL)
# viz(s,0)

print("done...")
    

fps = 5
arr = []

op = np.zeros(time)

for t in range(time):
    ss = s.copy()
    # arr.append(ss.reshape((Ly,Lx))) 
    if t == 10 or t == 100 or t == 1000 or t == 10000 or t == 100000 or t == int(1e6):
        viz_ising(s,t)
    
    op[t] = ordr_param(s)
    s = update_ising(s,beta)
    
plt.plot(op)
plt.ylim(0,1)
plt.show()

# def animate_func(i):
#     global im,arr
#     if i % fps == 0:
#         print( '.', end ='' )
    
#     xs,ys,vx,vy = pos_vel(arr[i],Lx,Ly)   
    
    
#     im.set_UVC(vx,vy)
#     im.set_offsets(np.array([ys,xs]).T)
    
#     time_text.set_text(f'Time: {i:n}')
#     return im,


# fig, ax = plt.subplots(figsize=(8, 16))
# x,y,v1,v2 = pos_vel(arr[0],Lx,Ly)

# im = plt.quiver(x,y,v1,v2,scale=Ly)
# ax.set_xlim(0, Lx)
# ax.set_ylim(0, Ly)
# time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
#                     bbox=dict(facecolor='white', alpha=0.75))
# plt.tight_layout()
# anim = animation.FuncAnimation(fig, animate_func, frames = time, 
#                                 interval = 1000 / fps)

# anim.save('q='+str(q)+'_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+'_time='+
#           str(time)+'.mp4', fps=fps)


print("Execution time:",datetime.now() - startTime)

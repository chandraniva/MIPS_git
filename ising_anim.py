import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

startTime = datetime.now()

red_shades = ["#FF0000", "#FF8080"]  # Two shades of red
blue_shades = ["#0000FF", "#8080FF"]  # Two shades of blue

# Create the colormap
colors = [red_shades[0]] + [blue_shades[0]]+ [red_shades[1]] + [blue_shades[1]]
cmap = ListedColormap(colors)

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
    return -2*eng
 

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
    
    
def viz_ising(s,t,desc):
    s[np.where(s>0)[0]] = 1
    s = s.reshape(Ly,Lx)
    fig = plt.figure()
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(1,1,1, aspect=2)
    ax1.imshow(s,cmap='binary')
    plt.title(desc+'  time='+str(t))
    # plt.savefig('is_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+
    #             '_time='+str(t)+'.png',dpi=500)
    plt.show()
    


def viz_circles(s,t,desc):
    s = s.reshape(Ly,Lx)
    fig = plt.figure()
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1, aspect=1)
    ax.set_xlim(-0.5, Lx-0.5)
    ax.set_ylim(-0.5, Ly-0.5)
    positions = np.where(s>0)
    if desc=='active':
        scatter = ax.scatter(positions[1], positions[0], s=50, 
                             c=s[positions], cmap=cmap, edgecolor='black')
        plt.title('HRTP: time = '+str(t))
        plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
        plt.savefig(desc+'_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+
                  '_time='+str(t)+'.png',dpi=500)
        
        
    else:
        scatter = ax.scatter(positions[1], positions[0], s=50, 
                             facecolor='gray', edgecolor='black')
        plt.title('Passive: time = '+str(t))
        plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
        plt.savefig(desc+'_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+
                  '_time='+str(t)+'.png',dpi=500)
    plt.show()



Lx, Ly = 32, 64
rho = 1/2
N = int(rho*Lx*Ly)
sqL = Lx*Ly
temp = 1.0
beta = 1/temp
time = 1000001
q = 0.1

for k in range(sqL):
    nbr=nbr2d(k,Lx,Ly)
nbr = nbr.astype('int16')

si = init_ord(N,sqL)
sq = init_ord(N,sqL)
    

fps = 20
arri = []
arrq = []

op = np.zeros(time)
opq = np.zeros(time)


for t in range(time):
    # ssi = si.copy()
    # ssq = sq.copy()
    # ssq[np.where(ssq>0)] = 1
    # ssi[np.where(ssi>0)] = 1
    # arri.append(ssi.reshape((Ly,Lx))) 
    # arrq.append(ssq.reshape((Ly,Lx))) 
    if t==0 or t == 100 or t==200 or t==300 or t == 1000 \
        or t == 10000 or t == 100000 or t == 200000 or t == 300000 or \
            t == 400000 or t == 500000 or  t == int(1e6):
        viz_circles(si,t,"ising")
        viz_circles(sq,t,"active")
    
    op[t] = ordr_param(si)
    opq[t] = ordr_param(sq)
    si = update_ising(si,beta)
    sq = update_q(sq,beta,q)
    
print("done...")

plt.plot(op,label='passive')
plt.plot(opq,label='active')
plt.legend()
plt.ylim(0,1)
plt.xscale('log')
plt.show()

# fig = plt.figure(figsize = (15,15))
# ax1 = fig.add_subplot(1,2,1, aspect=1)
# ax2 = fig.add_subplot(1,2,2, aspect=1)
# ax1.imshow(ssi.reshape(Ly,Lx),cmap='binary')
# ax2.imshow(ssq.reshape(Ly,Lx),cmap='binary')
# ax1.set_title("Passive Lattice gas",fontsize = 20)
# ax2.set_title("Motile lattice gas", fontsize = 20)
# time_text = ax1.text(1, 1.10, '', transform=ax1.transAxes, 
#                       fontsize=20, bbox=dict(facecolor='white', alpha=0.75))

# a = arri[0]
# b = arrq[0]
# im1 = ax1.imshow(a, cmap='binary')
# im2 = ax2.imshow(b, cmap='binary')



# def animate_func(i):
#     if i % fps == 0:
#         print( '.', end ='' )

#     im1.set_array(arri[i])    
#     im2.set_array(arrq[i])

#     time_text.set_text(f'Time: {i:n}')
#     return [im1, im2]

# anim = animation.FuncAnimation(fig, animate_func, frames = time, #blit = True, 
#                                 interval = 1000 / fps)
# # plt.tight_layout()

# anim.save('is_q_lx='+str(Lx)+'_ly='+str(Ly)+'_T='+str(temp)+'_time='+str(time)+
#           '.mp4', fps=fps)

# plt.show()

print("Execution time:",datetime.now() - startTime)

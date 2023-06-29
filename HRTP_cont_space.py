import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from math import ceil
from datetime import datetime
from numba.typed import List
from tqdm import tqdm
import math


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def build_cell_list(positions, L, r, N):
    cell_size = r
    n_cells = int(ceil(L / cell_size))
    max_particles_per_cell = N
    cell_list = np.full((n_cells * n_cells, max_particles_per_cell), -1, dtype=np.int64)
    cell_particle_counts = np.zeros(n_cells * n_cells, dtype=np.int64)
    cell_index = np.empty(N, dtype=np.int64)

    for i in prange(N):
        cell_x = int(positions[i, 0] // cell_size)
        cell_y = int(positions[i, 1] // cell_size)
        idx = int(cell_y * n_cells + cell_x)
        cell_index[i] = idx
        cell_list[idx, cell_particle_counts[idx]] = i
        cell_particle_counts[idx] += 1

    return cell_list, cell_particle_counts, n_cells


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def find_neighbors(positions, L, r, cell_list, cell_particle_counts, n_cells, N):
    max_neighbors = N  # Estimate max number of neighbors
    neighbors = np.empty((N, max_neighbors), dtype=np.int64)
    neighbor_counts = np.zeros(N, dtype=np.int64)
    r2 = r * r

    for i in prange(N):
        x, y = positions[i]
        cell_x = int(x // r)
        cell_y = int(y // r)
        neighbors[i, 0] = i
        count = 1
        adjlist = List()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                adj_x = (cell_x + dx) % n_cells
                adj_y = (cell_y + dy) % n_cells
                adj_idx = int(adj_y * n_cells + adj_x)
                adjlist.append(adj_idx) # if adj_idx not in adjlist else None

        for adj_idx in set(adjlist):
                for k in range(cell_particle_counts[adj_idx]):
                    j = cell_list[adj_idx, k]
                    if i != j: # Don't count self again
                        dx = abs(positions[j, 0] - x) % L
                        dy = abs(positions[j, 1] - y) % L
                        dx = min(dx, L - dx)
                        dy = min(dy, L - dy)

                        dist2 = dx * dx + dy * dy
                        if dist2 <= r2:
                            neighbors[i, count] = j
                            count += 1
        neighbor_counts[i] = count

    return neighbors, neighbor_counts


@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def H(i,positions,nbi):
    p1x = positions[0,i]
    p1y = positions[1,i]
    for j in nbi:
        dist = ((p1x - positions[0,j]) ** 2 + (p1y - positions[1,j]) ** 2) ** 0.5
        if dist > 0.5 and dist < 1:
            return -1
        elif dist > 1:
            return 0
        elif dist < 0.5:
            return 1000000
        

def viz(positions,directions):
    plt.figure(figsize=(8, 8))
    plt.xlim(0, L)
    plt.ylim(0, L)
    # for i in range(len(directions)):
    #     plt.arrow(
    #         positions[0,i], positions[1,i],
    #         0.5 * math.cos(directions[i]), 0.5 * math.sin(directions[i]),
    #         head_width=0.5, head_length=0.5, fc='r', ec='r'
    #     )
    scatter = plt.scatter(positions[1], positions[0], s=150,facecolor='gray', edgecolor='black')
    plt.show()



@jit(nopython=True, nogil=True, parallel=True, fastmath=True)
def update_HRTPcont(positions,directions,beta,q):
    N = len(directions)
    step = 0.25
    for _ in range(N):
        cell_list, cell_particle_counts, n_cells = build_cell_list(positions.T, L, r, N)
        neighbors, neighbor_counts = find_neighbors(positions.T, L, r, cell_list, cell_particle_counts, n_cells, N)
        
        i= int(np.random.rand()*N)
        x = positions[0, i]
        y = positions[1, i]
        nbi = [neighbors[i, j] for j in range(neighbor_counts[i])]
        if np.random.rand()<0.5: #tumble
            if np.random.rand()<q: 
                directions[i] = np.random.uniform(-np.pi,np.pi)
        else: #run
            eng_i = H(i,positions,nbi)
            positions[0, i] = (x + step* np.cos(directions[i])) % L
            positions[1, i] = (y + step* np.sin(directions[i])) % L
            eng_f = H(i,positions,nbi)
            if eng_f > eng_i:
                if np.random.rand()<1-np.exp(-beta*(eng_f-eng_i)):
                    positions[0, i] = x
                    positions[1, i] = y


    return positions, directions



# Lx, Ly = 32, 64
L = 50
Lx = Ly = L
rho = 0.5
N = int(rho*Lx*Ly)
sqL = Lx*Ly
r = 1
temp = 1.0
beta = 1/temp
q = 0.1
time = 100

print("Number of particles =",N)

xs = np.random.rand(N) * L
ys = np.random.rand(N) * L
ths = np.random.uniform(-np.pi, np.pi, size=N)
arr = np.empty((time + 1, 3, N))
arr[0] = xs, ys, ths
pos = np.array([xs, ys])


for t in tqdm(range(time)):
    # viz(pos,ths)
    pos, ths = update_HRTPcont(arr[t, :2, :], arr[t, 2, :],beta,q)
    arr[t + 1] = pos[0], pos[1], ths    
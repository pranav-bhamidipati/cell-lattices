import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy import sparse
from numba import jit
from functools import reduce
from itertools import combinations
import networkx
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


@jit(nopython=True)
def get_hex_neighbours(nx,ny):
    """
    Given the number of rows and columns, ny and nx, get the 6 neighbours of each cell.

    nc = number of cells
    i = 0,1,2,3,...,nc-1
    neighbours = of size nc x 6, the 6 neighbours for each i.

    Accounts for PBCS.
    :param nx:
    :param ny:
    :return:
    """
    nc = nx*ny
    i = np.arange(nc)
    neighbours = np.mod(np.column_stack(((i-nx,i-nx+1,i-1,i+1,i+nx,i+nx+1))),nc)
    return nc,i,neighbours

@jit(nopython=True)
def roll_axis1(M,i,ny):
    """
    Jitted version of roll axis = 1.
    M is the matrix. i is the roll number, and ny is the size of the 2nd dimension.
    :param M:
    :param i:
    :param ny:
    :return:
    """
    Mnew = M.copy()
    if i != 0:
        for j in range(ny):
            Mnew[j] = np.roll(M[j],i)
    return Mnew

@jit(nopython=True)
def get_translation_and_mirror_symmetries(state,nx,ny,nc):
    """
    Enumerates all possible transformations that preserve the pattern.

    I.e. nx * ny translations (shifting 0,1,2..., in the x direction, and in the y direction)
    And for each, considering the lr, ud and lr+ud reflections.
    :param state:
    :param nx:
    :param ny:
    :param nc:
    :return:
    """
    state_mat = state.reshape(nx,ny)
    trans_states_x = np.zeros((nx,nx,ny))
    for i in range(nx):
        trans_states_x[i] = roll_axis1(state_mat,i,ny)
    trans_states = np.zeros((nc,nx,ny),dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            trans_states[nx*j+i] = roll_axis1(trans_states_x[i].T,j,nx).T
    mirror_and_trans_states = np.zeros((nc*4,nc),dtype=np.int64)
    mirror_and_trans_states[::4] = trans_states.reshape(nc,nc)
    for i in range(nc):
        lr = np.fliplr(trans_states[i])
        mirror_and_trans_states[1 + i * 4] = lr.ravel()
        mirror_and_trans_states[2+i*4] = np.flipud(trans_states[i]).ravel()
        mirror_and_trans_states[3+i*4] = np.flipud(lr).ravel()
    return mirror_and_trans_states

def state_to_bin_hash(state):
    """
    Converts a state vector e.g. np.array((0,0,1,1,...)) to a string (reduce), then binarize, using int.
    :param state:
    :return:
    """
    return int(reduce(lambda x, y: x + str(y), state, ''),2)

def bin_hash_to_states(bin_hash,nc):
    """
    Converts a binary hash back to a state vector, the inverse of the above.

    nc = number of cells, important in padding. See below.
    :param bin_hash:
    :param nc:
    :return:
    """
    state_string = format(bin_hash,"b")
    state_string = "0"*(nc - len(state_string)) + state_string
    return np.array(list(state_string)).astype(np.int64)

def get_hash(state,nx,ny,nc):
    """
    Given a state vector, get the unique hash, accounting for all possible transformations.

    Unique hash = min(set of hashes under transformations).
    :param state:
    :param nx:
    :param ny:
    :param nc:
    :return:
    """
    all_states = get_translation_and_mirror_symmetries(state,nx,ny,nc)
    bin_hash = min(map(state_to_bin_hash,all_states))
    return bin_hash

# @jit(nopython=True)
def get_states_post_swap(_states,nx):
    """
    Get the state vectors post-swap.

    Two possible swaps are considered (a straight swap i.e. 0 <--> 1) and a diagonal swap (0 <--> nx).

    The idea is that, if we enumerate all possible states (i.e. like 5M if we have 12/25 red), and perform one of two
    swaps, then given all translations are accounted for in the hashing, this should sample ALL possible swaps of any state.

    Think this is correct, but don't have a proof.
    :param _states:
    :param nx:
    :return:
    """
    states = np.row_stack(_states)
    straight_swap = states.copy()
    diag_swap = states.copy()
    straight_swap[:,1],straight_swap[:,0] = states[:,0].copy(),states[:,1].copy()
    diag_swap[:,nx],diag_swap[:,0] = states[:,0].copy(),states[:,nx].copy()
    return straight_swap,diag_swap


@jit(nopython=True)
def get_state_from_combo(combo):
    """
    Combo is a tuple of cell ids, the "red" cells.
    Return a state vector, where 0 = blue, 1 = red.
    :param combo:
    :return:
    """
    state = np.zeros(25, dtype=np.int64)
    for c in combo:
        state[c] = 1
    return state

combos = combinations(np.arange(25),4) #with 4 red cells. Scale up at your own peril. 

states = list(map(get_state_from_combo,combos))

nx,ny,nc = 5,5,25

def _get_hash(state):
    """A wrapper for get_hash, so I can use the map function. """
    return get_hash(state, nx, ny, nc)


hashes = list(map(_get_hash,states)) #given the states, return the unique hashes upon all transformations
states_straight_swap,states_diag_swap = get_states_post_swap(states,nx) #get all of the post-swap states, in order.
hashes_straight_swap = list(map(_get_hash,list(states_straight_swap))) #return the unique hashes of each of the states after a "straight" swap (i.e. 0 for 1)
hashes_diag_swap = list(map(_get_hash,states_diag_swap))#and for diagonal swaps (i.e. 0 for nx)
print(len(set(hashes))) #counts the unique entries.
print(len(set(hashes_straight_swap)))
print(len(set(hashes_diag_swap))) #all the same, as expected

##with a full mesh, it will be infeasible to do on the RAM. But can save each hash one by one. Not currently built in -- Pranav??

unique_hashes = np.unique(hashes) #get unique hashes.
dictionary = dict(zip(unique_hashes,np.arange(unique_hashes.size))) #build a dictionary between the hash and the hash id (0,1,2,3,) so we can construct a matrix.
def call_hash_id(hash):
    return dictionary[hash]

hash_ids = list(map(call_hash_id,hashes)) #replaces hashes with hash_ids
hash_straight_ids = list(map(call_hash_id,hashes_straight_swap)) #and for each of the swap types.
hash_diag_ids = list(map(call_hash_id,hashes_diag_swap))

##Make a boolean matrix of size (n_unique_hashes x n_unique_hashes). matrix[i,j] = True if the pair of topologies can be made from each other by a single swap.
hash_conn = sparse.coo_matrix(([True]*(len(hash_ids)*2),
                               (hash_ids + hash_ids,hash_straight_ids+hash_diag_ids)),
                              shape=(len(unique_hashes),len(unique_hashes)))

##plot the connectivity matrix. just for a reference.
plt.imshow(hash_conn.toarray())
plt.show()


def _bin_hash_to_states(bin_hash):
    """another wrapper for a map function. """
    return bin_hash_to_states(bin_hash, nc)

##Get the "statate vector" of the unique hashes.
states_unique = list(map(_bin_hash_to_states,unique_hashes))

def get_cc_from_state(state):
    """
    Get the connectivity matrix from a state vector. Uses scipy function.
    Min value will be 2, as discussed.
    :param state:
    :return:
    """
    nc, i, neighbours = get_hex_neighbours(nx, ny)
    Is = np.repeat(i, 6)
    edges = np.column_stack((Is, neighbours.ravel()))
    state_edges = state[edges]
    same_edges = (state_edges[:,0] == state_edges[:,1])
    adj_mat = sparse.coo_matrix(([1]*sum(same_edges),(edges[same_edges,0],edges[same_edges,1])),shape = (nc,nc))
    ncc,__ = sparse.csgraph.connected_components(adj_mat,directed=False)
    return ncc

cc_unique = list(map(get_cc_from_state,states_unique)) #get number of connected components for each unique state (i.e. each row/col of the adj matrix)


##Plot the network of topolgoies
G = networkx.convert_matrix.from_scipy_sparse_matrix(hash_conn)
pos = networkx.spring_layout(G,iterations=500,dim=3)##use spring layout to get the positions of each node.
pos = np.array(list(pos.values()))

##Use plotly to build the plot.
edge_x = []
edge_y = []
edge_z = []
indices = sparse.csr_matrix(hash_conn).nonzero()

for i,j in zip(*indices):
    edge_x.append(pos[i,0])
    edge_y.append(pos[i,1])
    edge_z.append(pos[i,2])
    edge_x.append(pos[j,0])
    edge_y.append(pos[j,1])
    edge_z.append(pos[j,2])
    edge_x.append(None)
    edge_y.append(None)
    edge_z.append(None)

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y,z=edge_z,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')


node_trace = go.Scatter3d(
    x=pos[:,0], y=pos[:,1],z=pos[:,2],
    mode='markers')
node_trace.marker.color = np.array(cc_unique)
node_trace.text = ["Number of connected components = %d"%cc for cc in np.array(cc_unique)]
fig = go.Figure(data=[edge_trace,node_trace])
fig.show()

"""
I'll keep annotating in the future, but will stop here :) 
"""

###Energies

nc, i, neighbours = get_hex_neighbours(nx, ny)
W = np.array(((1,3),
              (3,1)))
def get_energy_from(state):
    Is = np.repeat(i, 6)
    edges = np.column_stack((Is, neighbours.ravel()))
    state_edges = state[edges]
    energy = W.take(state_edges[:,0]+2*state_edges[:,1]).sum()
    return energy

energy_unique = np.array(list(map(get_energy_from,states_unique)))
dE = energy_unique - np.expand_dims(energy_unique,1)

Ea = np.abs(dE)*2 + dE
# T = 5
# Ea = dE



nt = 1000
dt = 1e-3
t_span = np.arange(0,dt*nt+dt,dt)
def simulate(Ea,T):
    pr_transition = np.exp(-Ea / T)
    pr_transition = np.clip(pr_transition, 0, 1)
    init_p = np.ones_like(unique_hashes) / len(unique_hashes)
    # init_p = np.random.uniform(0,1,len(unique_hashes))
    p = init_p.copy()
    p_save = np.zeros((nt+1, p.size))
    p_save[0] = p
    for i in range(1,nt+1):
        dtp = (pr_transition.T@p -pr_transition.sum(axis=1)*p)
        p = p + dt*dtp
        p_save[i] = p
    return p_save

#
# def weighted_std(values, weights):
#     """
#     Return the weighted average and standard deviation.
#
#     values, weights -- Numpy ndarrays with the same shape.
#     """
#     average = np.average(values, weights=weights)
#     # Fast and numerically precise:
#     variance = np.average((values-average)**2, weights=weights)
#     return np.sqrt(variance)

# p_save = simulate(Ea,10)
# average_no_con_comp = (np.array(cc_unique)*p_save).sum(axis=1)
# std_no_con_comp = np.array([weighted_std(np.array(cc_unique),p) for p in p_save])
#
# plt.fill_between(t_span,average_no_con_comp - std_no_con_comp,average_no_con_comp + std_no_con_comp,alpha=0.4)
# plt.plot(t_span,average_no_con_comp)
# plt.show()

# pr_transition = (pr_transition.T/pr_transition.sum(axis=1)).T

sorted_mask = np.array(cc_unique)==2
T_space = np.logspace(-1,1.5,30)
cols = plt.cm.plasma(np.linspace(0,1,T_space.size))
p_save_span= np.array([simulate(Ea,T) for T in T_space])
prop_sorted = p_save_span[...,sorted_mask].sum(axis=-1)
fig, ax = plt.subplots()
for i, T in enumerate(T_space):
    ax.plot(t_span[:300],prop_sorted[i][:300],color=cols[i])
fig.savefig("energy_analysis/energy_barrier.pdf",dpi=300)

fig, ax = plt.subplots()
ax.plot(np.log10(T_space),prop_sorted[:,-1])
fig.show()

def tanhfit(t,end,v0):
    m = 2*v0/end
    return prop_sorted[0,0] + end/2 * (1 + np.tanh(m*t))

from scipy.optimize import curve_fit

args,__ = curve_fit(tanhfit,t_span,prop_sorted[0])
ends,v0s = [],[]
for ps in prop_sorted:
    (end,v0),__ = curve_fit(tanhfit, t_span, ps)
    ends.append(end)
    v0s.append(v0)


fig, ax1 = plt.subplots(figsize=(4,3))
ax1.set(xlabel=r"$log_{10} \ T$")
ax1.set_ylabel(ylabel=r"$Pr(sorted | t \rightarrow \infty)$", color="tab:red")  # we already handled the x-label with ax1

ax1.plot(np.log10(T_space), ends, color="tab:red")
ax1.tick_params(axis='y', labelcolor="tab:red")
# ax1.set(ylim=(0,1))
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set(xlabel=r"$log_{10} \ T$",ylabel="v0") # we already handled the x-label with ax1
ax2.set_ylabel(r'$v_0$', color="tab:blue")  # we already handled the x-label with ax1
ax2.plot(np.log10(T_space), v0s, color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")
fig.subplots_adjust(bottom=0.3,top=0.8,left=0.3,right=0.8)
fig.savefig("energy_analysis/hex_sorting_v0_asymptote.pdf",dpi=300)



plt.plot(t_span,tanhfit(t_span,*args))
plt.plot(t_span,prop_sorted[-1])
plt.show()


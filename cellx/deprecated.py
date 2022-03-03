import numpy as np
import numba

from functools import reduce
from itertools import combinations


### Some binary hashing code
def binstr_to_state(binstr):
    """Converts a binary string to binary state vector."""
    return np.array([int(d) for d in binstr])

def binstr_to_bin_hash(binstr):
    """Converts binary string directly to binary hash (type integer)."""
    return state_to_bin_hash(binstr_to_state(binstr))

def binstr_to_indices(binstr):
    return binstr_to_state(binstr).nonzero()[0]

def indices_to_state(idx, n):
    arr = np.zeros(n, dtype=int)
    arr[np.asarray(idx)] = True
    return arr

def state_to_binstr(state):
    return "".join([str(i) for i in state])

def indices_to_binstr(idx, n):
    return state_to_binstr(indices_to_state(idx, n))

def indices_to_bin_hash(idx, n):
    return state_to_bin_hash(indices_to_state(idx, n))

##########################
####### Jake's code ######
##########################

# @numba.njit
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

# @numba.njit
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

# @numba.njit
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

# @numba.njit
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


# @numba.njit
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


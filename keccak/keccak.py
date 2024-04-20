# -*- coding: utf-8 -*-
"""
Python implementation of the keccak function and derived SHA-3 hash.

The round components and their inverse are implemented, along with complete
SHA3 instances.

Dependencies:
    - numpy, used to represent the internal state of the function.
    - sympy, used in the inversion of the `theta` round components.

References
----------
* https://keccak.team/index.html
* https://csrc.nist.gov/projects/hash-functions/sha-3-project
* SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions
  https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
* The Keccak reference
  https://csrc.nist.gov/csrc/media/Projects/hash-functions/documents/Keccak-reference-3.0.pdf
* https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/example-values
"""

import numpy as np
# import sympy as sp


# =============================================================================
# Setup
# =============================================================================

## theta inverse
# def compute_theta_inv(l: int = 6)-> dict:
#     """
#     Compute the rolls to apply to the parity in order to inverse theta.
#     """
#     w = 2**l
#     k = w * sp.mod_inverse(3*w, 5)
    
#     t = sp.Symbol('t')
#     Igen = sp.Poly(1 + t**(5*w), gens=(t,), domain=sp.GF(2))
#     P = sp.Poly(1 + t**k + t**(k+1), gens=(t,), domain=sp.GF(2))
#     Q = 1 + sp.invert(P, Igen, domain=sp.GF(2))
    
#     x, z = sp.Symbol('x'), sp.Symbol('z')
#     Q = sp.Poly(Q(x**3 * z), gens=(x, z))
#     Q = sp.rem(Q, x**5 + 1, gens=(x, z), domain=sp.GF(2)) # reduce mod 1 + x^5
#     Q = sp.rem(Q, z**w + 1, gens=(z, x), domain=sp.GF(2)) # reduce mod 1 + z^w
    
#     roll_inv = {}
#     for i in range(l+1):
#         R = sp.rem(Q, z**(2**i) + 1, gens=(z, x), domain=sp.GF(2))
#         roll_inv[i] = [(j, k) for (k, j) in R.as_dict()]
#     return roll_inv

# THETA_INV = compute_theta_inv(l=6)

THETA_INV = {
    1: [(0, 0), (2, 0), (4, 0), (0, 1), (3, 1), (4, 1)],
    2: [(3, 0), (2, 1), (3, 1), (0, 2), (2, 2),
        (3, 2), (4, 2), (0, 3), (2, 3), (4, 3)],
    3: [(2, 0), (0, 1), (4, 1), (0, 2), (1, 2), (2, 2), (4, 2),
        (0, 3), (3, 3), (2, 4), (3, 4), (0, 5), (2, 5), (3, 5),
        (4, 5), (1, 6), (3, 6), (2, 7), (3, 7), (4, 7)],
    4: [(0, 0), (1, 0), (4, 0), (3, 1), (4, 1), (0, 2), (1, 2), (3, 2), (4, 2),
        (2, 3), (4, 3), (1, 4), (2, 4), (1, 5), (2, 5), (3, 5), (4, 5), (0, 6),
        (2, 6), (0, 7), (4, 7), (0, 8), (1, 8), (2, 8), (4, 8), (0, 9), (3, 9),
        (2, 10), (3, 10), (0, 11), (2, 11), (3, 11), (4, 11), (1, 12), (3, 12),
        (0, 13), (1, 13), (0, 14), (1, 14), (2, 14), (3, 14), (0, 15), (2, 15),
        (3, 15)],
    5: [(0, 0), (2, 0), (4, 0), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (4, 2),
        (0, 3), (2, 3), (0, 4), (4, 4), (0, 5), (1, 5), (2, 5), (4, 5), (0, 6),
        (3, 6), (2, 7), (3, 7), (0, 8), (2, 8), (3, 8), (4, 8), (1, 9), (3, 9),
        (0, 10), (1, 10), (0, 11), (1, 11), (2, 11), (3, 11), (1, 12), (4, 12),
        (3, 13), (4, 13), (0, 14), (1, 14), (3, 14), (4, 14), (2, 15), (4, 15),
        (1, 16), (2, 16), (1, 17), (2, 17), (3, 17), (4, 17), (0, 18), (2, 18),
        (0, 19), (4, 19), (0, 20), (1, 20), (2, 20), (4, 20), (0, 21), (3, 21),
        (2, 22), (3, 22), (0, 23), (2, 23), (3, 23), (4, 23), (1, 24), (3, 24),
        (0, 25), (1, 25), (0, 26), (1, 26), (2, 26), (3, 26), (1, 27), (4, 27),
        (3, 28), (4, 28), (0, 29), (1, 29), (3, 29), (4, 29), (2, 30), (4, 30),
        (0, 31), (3, 31), (4, 31)],
    6: [(3, 0), (2, 1), (3, 1), (0, 2), (2, 2), (3, 2), (4, 2), (1, 3), (3, 3),
        (0, 4), (1, 4), (0, 5), (1, 5), (2, 5), (3, 5), (1, 6), (4, 6), (3, 7),
        (4, 7), (0, 8), (1, 8), (3, 8), (4, 8), (2, 9), (4, 9), (1, 10),
        (2, 10), (1, 11), (2, 11), (3, 11), (4, 11), (0, 12), (2, 12), (0, 13),
        (4, 13), (0, 14), (1, 14), (2, 14), (4, 14), (0, 15), (3, 15), (2, 16),
        (3, 16), (0, 17), (2, 17), (3, 17), (4, 17), (1, 18), (3, 18), (0, 19),
        (1, 19), (0, 20), (1, 20), (2, 20), (3, 20), (1, 21), (4, 21), (3, 22),
        (4, 22), (0, 23), (1, 23), (3, 23), (4, 23), (2, 24), (4, 24), (1, 25),
        (2, 25), (1, 26), (2, 26), (3, 26), (4, 26), (0, 27), (2, 27), (0, 28),
        (4, 28), (0, 29), (1, 29), (2, 29), (4, 29), (0, 30), (3, 30), (2, 31),
        (3, 31), (0, 32), (2, 32), (3, 32), (4, 32), (1, 33), (3, 33), (0, 34),
        (1, 34), (0, 35), (1, 35), (2, 35), (3, 35), (1, 36), (4, 36), (3, 37),
        (4, 37), (0, 38), (1, 38), (3, 38), (4, 38), (2, 39), (4, 39), (1, 40),
        (2, 40), (1, 41), (2, 41), (3, 41), (4, 41), (0, 42), (2, 42), (0, 43),
        (4, 43), (0, 44), (1, 44), (2, 44), (4, 44), (0, 45), (3, 45), (2, 46),
        (3, 46), (0, 47), (2, 47), (3, 47), (4, 47), (1, 48), (3, 48), (0, 49),
        (1, 49), (0, 50), (1, 50), (2, 50), (3, 50), (1, 51), (4, 51), (3, 52),
        (4, 52), (0, 53), (1, 53), (3, 53), (4, 53), (2, 54), (4, 54), (1, 55),
        (2, 55), (1, 56), (2, 56), (3, 56), (4, 56), (0, 57), (2, 57), (0, 58),
        (4, 58), (0, 59), (1, 59), (2, 59), (4, 59), (0, 60), (3, 60), (2, 61),
        (3, 61), (0, 62), (2, 62), (3, 62), (4, 62), (0, 63), (2, 63), (4, 63)]
    }


## chi inverse lookup table
def compute_chi_inv():
    """
    Compute the inverse of the non-linear keccak component chi as a lookup
    table.
    """
    states = np.array([[(x >> i) & 1 for i in range(5)] for x in range(2**5)],
                      dtype=bool)
    chi_states = states ^ (~np.roll(states, (0, -1), axis=(0, 1))
                           & np.roll(states, (0, -2), axis=(0, 1)))
    chi_idx = np.sum(chi_states * 2**np.arange(5), axis=1)
    return states[np.argsort(chi_idx)]

CHI_INV = compute_chi_inv()


## iota shifts
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
      0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
      0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
      0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
      0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
      0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
      0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
RC = np.array([[(r >> i) & 1 for i in range(64)] for r in RC], dtype=bool)

# =============================================================================
# Keccak components
# =============================================================================

########## Round components ##########

def theta(state: np.ndarray)-> np.ndarray:
    """
    theta permutation of keccak-f[5*5*2**l]
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    parity = np.bitwise_xor.reduce(state, axis=0)
    return (state
            ^ np.roll(parity, (1, 0), axis=(0, 1))
            ^ np.roll(parity, (-1, 1), axis=(0, 1)))


def rho(state: np.ndarray)-> np.ndarray:
    """
    rho permutation of keccak-f[5*5*2**l]
    Diffusion along k.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    M = np.array([[3, 2], [1, 0]], dtype=int)
    new_state = np.empty_like(state)
    new_state[0, 0] = state[0, 0]
    ij = np.array([0, 1])
    for t in range(24):
        new_state[*ij] = np.roll(state[*ij], (t+1)*(t+2) // 2)
        ij = (M @ ij) % 5
    return new_state


def pi(state: np.ndarray)-> np.ndarray:
    """
    pi permutation of keccak-f[5*5*2**l]
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    M = np.array([[0, 1], [3, 1]], dtype=int)
    ij_ = np.einsum("ij,jkl->ikl", M, np.mgrid[0:5, 0:5]) % 5
    return state[*ij_]
    

def chi(state: np.ndarray)-> np.ndarray:
    """
    chi permutation of keccak-f[5*5*2**l]
    The non-linear operation.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    return state ^ (~np.roll(state, (0, -1, 0), axis=(0, 1, 2))
                    & np.roll(state, (0, -2, 0), axis=(0, 1, 2)))
    


def iota(state: np.ndarray, round_number: int)-> np.ndarray:
    """
    pi permutation of keccak-f[5*5*2**l]
    Breaks translation invariance along axis 2.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    state[0, 0] ^= RC[round_number][:state.shape[-1]]
    return state


########## Inverse Round components ##########

def inv_theta(state: np.ndarray)-> np.ndarray:
    """
    inverse theta permutation of keccak-f[5*5*2**l]
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    l = state.shape[-1].bit_length() - 1
    parity = np.bitwise_xor.reduce(state, axis=0)
    inv = np.zeros_like(parity)
    for jk in THETA_INV[l]:
        inv ^= np.roll(parity, jk, axis=(0, 1))
    return state ^ inv
    

def inv_rho(state: np.ndarray)-> np.ndarray:
    """
    chi permutation of keccak-f[5*5*2**l]
    The non-linear operation.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    M = np.array([[3, 2], [1, 0]], dtype=int)
    inv = np.empty_like(state)
    inv[0, 0] = state[0, 0]
    ij = np.array([0, 1])
    for t in range(24):
        inv[*ij] = np.roll(state[*ij], -(t+1)*(t+2) // 2)
        ij = (M @ ij) % 5
    return inv


def inv_pi(state: np.ndarray)-> np.ndarray:
    """
    Inverse pi permutation of keccak-f[5*5*2**l]
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    M = np.array([[3, 2], [1, 0]], dtype=int)
    ij_ = np.einsum("ij,jkl->ikl", M, np.mgrid[0:5, 0:5]) % 5
    return state[*ij_]


def inv_chi(state: np.ndarray)-> np.ndarray:
    """
    chi permutation of keccak-f[5*5*2**l]
    The non-linear operation.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    state_idx = np.sum(state * 2**np.arange(5)[:, None], axis=1)
    inv = CHI_INV[state_idx]
    return np.swapaxes(inv, 1, 2)


def inv_iota(state: np.ndarray, round_number: int)-> np.ndarray:
    """
    pi permutation of keccak-f[5*5*2**l]
    Breaks translation invariance along axis 2.
    
    state : np.ndarray[bool] of shape (5, 5, 2**l)
    """
    state[0, 0] ^= RC[round_number][:state.shape[-1]]
    return state


########## Keccak round function ##########

def keccak_round(state: np.ndarray, round_number: int)-> np.ndarray:
    """
    Full keccak round function.
    """
    s = theta(state)
    s = rho(s)
    s = pi(s)
    s = chi(s)
    s = iota(s, round_number)
    return s


def inv_keccak_round(state: np.ndarray, round_number: int)-> np.ndarray:
    """
    Inverse of keccak round function.
    """
    s = inv_iota(state, round_number)
    s = inv_chi(s)
    s = inv_pi(s)
    s = inv_rho(s)
    s = inv_theta(s)
    return s


# =============================================================================
# SHA3
# =============================================================================

def configure_hash_func(sha3_instance: str, output_size: int)-> tuple[int, ...]:
    """
    Set the configuration of the selected SHA3 instance.
    """
    l = 6
    n = 12 + 2*l
    if sha3_instance.upper() == "SHA3-224":
        d, r, suffix = 224, 1152, [0, 1]
    elif sha3_instance.upper() == "SHA3-256":
        d, r, suffix = 256, 1088, [0, 1]
    elif sha3_instance.upper() == "SHA3-384":
        d, r, suffix = 384, 832, [0, 1]
    elif sha3_instance.upper() == "SHA3-512":
        d, r, suffix = 512, 576, [0, 1]
    elif sha3_instance.upper() == "SHAKE128":
        d, r, suffix = output_size, 1344, [1, 1, 1, 1]
    elif sha3_instance.upper() == "SHAKE256":
        d, r, suffix = output_size, 1088, [1, 1, 1, 1]
    else:
        msg = "hash_type must be'SHA3-224', 'SHA3-256', 'SHA3-384', 'SHA3-512' "
        msg += "'SHAKE128' or 'SHAKE256'"
        raise ValueError(msg)
    return 25*2**l, n, d, r, suffix


def format_input(data: bytes, suffix: list[int], rate: int)-> np.ndarray:
    """
    Format bytes into array.
    """
    data = np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                         bitorder='little')
    size = len(data) + len(suffix) + 2
    nblocks = size // rate + (size % rate > 0)
    
    blocks = np.zeros((nblocks*rate,), dtype=bool)
    blocks[:len(data)] = data
    blocks[len(data):len(data)+len(suffix)] = suffix
    blocks[len(data)+len(suffix)] = 1
    blocks[-1] = 1
    
    return blocks.reshape((-1, rate))


def sha3(data: bytes,
         sha3_instance: str = "SHA3-512",
         output_size: int = None)-> bytes:
    """
    Hash the data with selected SHA3 instance.
    """
    sz, n, d, r, suffix = configure_hash_func(sha3_instance, output_size)
    
    state = np.zeros((sz,), dtype=bool)
    input_blocks = format_input(data, suffix, r)
    # absorption
    for block in input_blocks:
        state[:r] ^= block
        state = state.reshape((5, 5, -1))
        for i in range(n):
            state = keccak_round(state, i)
        state = state.flatten()
    # squeezing
    hash_val = state[:r].tolist()
    while len(hash_val) < d:
        state = state.reshape((5, 5, -1))
        for i in range(n):
            state = keccak_round(state, i)
        state = state.flatten()
        hash_val += state[:r].tolist()
    
    return np.packbits(hash_val[:d], bitorder='little').tobytes()
    

def inv_sha3(final_state: bytes,
             sha3_instance: str = "SHA3-512",
             output_size: int = None)-> bytes:
    """
    Reverse keccak on final_state to recover the input.
    The final state must be the whole state, not just the digest.
    """
    sz, n, d, r, suffix = configure_hash_func(sha3_instance, output_size)
    
    state = format_input(final_state, [], sz)[0].reshape((5, 5, -1))
    for i in range(n-1, -1, -1):
        state = inv_keccak_round(state, i)
    state = state.flatten()
    msg = np.packbits(state[:r], bitorder='little').tobytes()
    return msg


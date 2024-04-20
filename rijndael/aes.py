# -*- coding: utf-8 -*-
"""
Python implementation of the AES symmetric cipher.

Are implemented:
    - AES round components and their inverse
    - AES key schedule and key recovery from expanded key
    - AES encryption/decryption (only in ECB mode)

References
----------
* FIPS 197-upd1: Advanced Encryption Standard (AES)
  https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf
* https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/example-values
"""

import numpy as np


# =============================================================================
# GF(2^8) algebra
# =============================================================================

def mult_table(factor: int)-> np.ndarray:
    """
    Compute the multiplication table x * factor.
    """
    p = 0b100011011 # irreducible polynomial x^8 + x^4 + x^3 + x + 1
    x = np.arange(256, dtype=np.uint16)
    m = np.zeros(256, dtype=np.uint16)
    for i in range(7, -1, -1): # carryless multiply
        if (factor >> i) & 1:
            m ^= x << i
    for i in range(7, -1, -1): # reduce modulo p
        m[np.nonzero((m >> (i+8)) & 1)] ^= p << i
    return m.astype(np.uint8)


def inverse_table()-> np.ndarray:
    """
    Compute the table of inverses of x.
    """
    p = 0b100011011 # irreducible polynomial x^8 + x^4 + x^3 + x + 1
    # prepare list of powers x^(2^k)
    x = np.zeros((8, 256), dtype=np.uint16)
    x[0] = np.arange(256, dtype=np.uint16)
    for k in range(1, 8):
        for i in range(7, -1, -1): # square
            shift = np.nonzero((x[k-1] >> i) & 1)
            x[k, shift] ^= x[k-1, shift] << i
        for i in range(7, -1, -1): # reduce modulo p
            x[k, np.nonzero((x[k] >> (i+8)) & 1)] ^= p << i
    # multiply x^(2+4+8+16+32+64+128) = x^254 = 1/x
    for k in range(2, 8):
        inv = np.zeros(256, dtype=np.uint16)
        for i in range(7, -1, -1): # multiply
            shift = np.nonzero((x[k-1] >> i) & 1)
            inv[shift] ^= x[k, *shift] << i
        for i in range(7, -1, -1): # reduce modulo p
            inv[np.nonzero((inv >> (i+8)) & 1)] ^= p << i
        x[k] = inv
    return inv.astype(np.uint8)
    

# =============================================================================
# AES setup
# =============================================================================

def s_box()-> np.ndarray:
    """
    Compute Rijndael S-box.
    """
    b = np.unpackbits(inverse_table()[:, None], axis=1, bitorder='big')
    s = np.copy(b)
    for i in range(1, 5):
        s ^= np.roll(b, (0, -i), axis=(0, 1))
    s = np.packbits(s, axis=1, bitorder='big').flatten()
    s ^= 0x63
    return s

S_BOX = s_box()
INV_S_BOX = np.argsort(S_BOX).astype(np.uint8)


def mix_cols(mult: tuple[int, ...])-> np.ndarray:
    """
    Compute Rikndael MixColumns operation.
    """
    mix = np.empty((len(mult), len(mult), 256), dtype=np.uint8)
    mix[0] = [mult_table(m) for m in mult]
    for i in range(1, len(mult)):
        mix[i] = np.roll(mix[0], i, axis=0)
    return mix

MIX_COLS = mix_cols((2, 3, 1, 1))
INV_MIX_COLS = mix_cols((14, 11, 13, 9))

# =============================================================================
# AES components
# =============================================================================

########## Round components ##########

def s_box(arr: np.ndarray):
    """
    S-box substitution to all bytes of the array.
    """
    return S_BOX[arr]


def shift_rows(state: np.ndarray):
    """
    ShiftRows procedure.
    """
    for j in range(1, 4):
        state[..., j] = np.roll(state[..., j], -j, axis=-1)


def mix_columns(state: np.ndarray):
    """
    MixColumns procedure.
    """
    sh, (shi, shj) = state.shape[:-2], state.shape[-2:]
    for i in range(shi):
        s = np.zeros(sh + (shj,), dtype=np.uint8)
        for j in range(shj):
            mc = MIX_COLS[j, np.arange(shj), state[..., i, :]]
            s[..., j] = np.bitwise_xor.reduce(mc, axis=-1)
        state[..., i, :] = s


########## Inverse Round components ##########

def inv_s_box(arr: np.ndarray):
    """
    Inverse S-box substitution to all bytes of the array.
    """
    return INV_S_BOX[arr]


def inv_shift_rows(state: np.ndarray):
    """
    Inverse ShiftRows procedure.
    """
    for j in range(1, 4):
        state[..., j] = np.roll(state[..., j], j, axis=-1)


def inv_mix_columns(state: np.ndarray):
    """
    Inverse MixColumns procedure.
    """
    sh, (shi, shj) = state.shape[:-2], state.shape[-2:]
    for i in range(shi):
        s = np.zeros(sh + (shj,), dtype=np.uint8)
        for j in range(shj):
            mc = INV_MIX_COLS[j, np.arange(shj), state[..., i, :]]
            s[..., j] = np.bitwise_xor.reduce(mc, axis=-1)
        state[..., i, :] = s


########## Key expansion ##########

def key_expansion(key: np.ndarray, nb_rkeys: int)-> np.ndarray:
    """
    AES key schedule
    """
    # setup round constants
    p = 0b100011011 # irreducible polynomial x^8 + x^4 + x^3 + x + 1
    rcon = np.zeros((10, 4), dtype=np.uint16)
    rcon[0, 0] = 1
    for i in range(1, len(rcon)):
        rcon[i, 0] = rcon[i-1, 0] << 1
        if rcon[i, 0] >> 8 & 1:
            rcon[i, 0] ^= p
    # key schedule
    N = key.size // 4
    W = np.empty((4*nb_rkeys, 4), dtype=np.uint8)
    W[:N] = key.reshape((N, 4))
    for i in range(N, 4*nb_rkeys):
        if i % N == 0:
            W[i] = W[i-N] ^ s_box(np.roll(W[i-1], -1)) ^ rcon[i//N-1]
        elif N > 6 and i % N == 4:
            W[i] = W[i-N] ^ s_box(W[i-1])
        else:
            W[i] = W[i-N] ^ W[i-1]
    return W


def inv_key_expansion(rkey: np.ndarray, R: int)-> np.ndarray:
    """
    Inverse AES key schedule from a 'complete' round key that terminates at
    the beginning of round R (ie if rkey is the last round key then
    R = nb rounds + 1).
    
    Complete means that the length of the provided round key is the same as
    that of the key.

    Parameters
    ----------
    rkey : np.ndarray[np.uint8] of shape (n,) or (4, n//4) with n = 16, 24, 32
        Round key.
    R : int
        Round number.
    """
    # setup round constants
    p = 0b100011011 # irreducible polynomial x^8 + x^4 + x^3 + x + 1
    rcon = np.zeros((10, 4), dtype=np.uint16)
    rcon[0, 0] = 1
    for i in range(1, len(rcon)):
        rcon[i, 0] = rcon[i-1, 0] << 1
        if rcon[i, 0] >> 8 & 1:
            rcon[i, 0] ^= p
    # reverse key schedule
    N = rkey.size // 4
    W = np.empty((4*R, 4), dtype=np.uint8)
    W[-N:] = rkey.reshape((N, 4))
    
    for i in range(4*R-N-1, -1, -1):
        if i % N == 0:
            W[i] = W[i+N] ^ s_box(np.roll(W[i+N-1], -1)) ^ rcon[i//N]
        elif N > 6 and i % N == 4:
            W[i] = W[i+N] ^ s_box(W[i+N-1])
        else:
            W[i] = W[i+N] ^ W[i+N-1]
    return W


def key_retrieval(rkey: np.ndarray, R: int)-> np.ndarray:
    """
    Retrieve the key from a complete round key.
    Complete means that the length of the provided round key is the same as
    that of the key.
    """
    N = len(rkey) // 4
    W = inv_key_expansion(rkey, R)
    return W[:N].flatten()


# =============================================================================
# AES
# =============================================================================

def block_crypt(block: np.ndarray,
                round_keys: np.array,
                nb_rounds: int)-> np.ndarray:
    """
    Block AES encryption.

    Parameters
    ----------
    block : np.ndarray[np.uint8] of shape (16,)
    round_keys : np.array[np.uint8] of shape (4*nb_rounds+1, 4)
    nb_rounds : int = 10 (AES 128), 12 (AES 192), 14 (AES 256)
    """
    cipher = np.copy(block.reshape((4, 4)))
    round_keys = round_keys.reshape((-1, 4, 4))
    # Initial round key addition
    cipher ^= round_keys[0]
    # bulk rounds
    for i in range(1, nb_rounds):
        cipher = s_box(cipher)
        shift_rows(cipher)
        mix_columns(cipher)
        cipher ^= round_keys[i]
    # final round
    cipher = s_box(cipher)
    shift_rows(cipher)
    cipher ^= round_keys[nb_rounds]
    return cipher.flatten()


def block_decrypt(block: np.ndarray,
                  round_keys: np.array,
                  nb_rounds: int)-> np.ndarray:
    """
    Block AES decryption.

    Parameters
    ----------
    block : np.ndarray[np.uint8] of shape (16,)
    round_keys : np.array[np.uint8] of shape (4*nb_rounds+1, 4)
    nb_rounds : int 10 (AES 128), 12 (AES 192), 14 (AES 256)
    """
    plain = np.copy(block.reshape((4, 4)))
    round_keys = round_keys.reshape((-1, 4, 4))
    # Undo final round
    plain ^= round_keys[nb_rounds]
    inv_shift_rows(plain)
    plain = inv_s_box(plain)
    # Undo bulk rounds
    for i in range(nb_rounds-1, 0, -1):
        plain ^= round_keys[i]
        inv_mix_columns(plain)
        inv_shift_rows(plain)
        plain = inv_s_box(plain)
    # Undo initial round key addition
    plain ^= round_keys[0]
    return plain.flatten()


def configure_aes(key: bytes)-> int:
    """
    Return the number of AES rounds corresponding to the given key size.
    16 bytes -> 10 rounds (AES-128)
    24 bytes -> 12 rounds (AES-192)
    32 bytes -> 14 rounds (AES-256)
    Raises ValueError otherwise.
    """
    if len(key) == 16:
        return 10
    elif len(key) == 24:
        return 12
    elif len(key) == 32:
        return 14
    msg = b"key length must be 128, 192 or 256 bits, got {8*len(key)}"
    raise ValueError(msg)


def format_input(text: bytes)-> np.ndarray:
    """
    Convert a byte-encoded message to blocks suitable for AES
    encryption/decryption.
    """
    blocks = np.frombuffer(text, dtype=np.uint8).reshape((-1, 16))
    return blocks


def aes_crypt(clear: bytes, key: bytes, mode: str = 'ECB')-> bytes:
    """
    Encrypt with AES.

    Parameters
    ----------
    clear : bytes
        The byte-encoded data to encrypt.
    key : bytes
        The AES key. The AES instance is selected according to its length:
            - 16 bytes -> AES-128
            - 24 bytes -> AES-192
            - 32 bytes -> AES-256
    mode : str, optional
        Encryption mode: only ECB (Encryption CodeBook) mode is implemented.
        The default is 'ECB'.
    """
    nb_rounds = configure_aes(key)
    key = np.frombuffer(key, dtype=np.uint8)
    round_keys = key_expansion(key, nb_rounds + 1)
    blocks = format_input(clear)
    cipher = np.empty_like(blocks)
    for i, block in enumerate(blocks):
        cipher[i] = block_crypt(block, round_keys, nb_rounds)
    return cipher.flatten().tobytes()


def aes_decrypt(cipher: bytes, key: bytes, mode: str = 'ECB')-> bytes:
    """
    Decrypt AES-encrypted data.

    Parameters
    ----------
    cipher : bytes
        Data to decrypt.
    key : bytes
        The AES key. The AES instance is selected according to its length:
            - 16 bytes -> AES-128
            - 24 bytes -> AES-192
            - 32 bytes -> AES-256
    mode : str, optional
        Encryption mode: only ECB (Encryption CodeBook) mode is implemented.
        The default is 'ECB'.
    """
    nb_rounds = configure_aes(key)
    key = np.frombuffer(key, dtype=np.uint8)
    round_keys = key_expansion(key, nb_rounds + 1)
    blocks = format_input(cipher)
    plain = np.empty_like(blocks)
    for i, block in enumerate(blocks):
        plain[i] = block_crypt(block, round_keys, nb_rounds)
    return plain.flatten().tobytes()

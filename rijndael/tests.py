# -*- coding: utf-8 -*-
"""
Tests for the aes module.
"""

import numpy as np

from aes import (s_box, shift_rows, mix_columns,
                 inv_s_box, inv_shift_rows, inv_mix_columns,
                 key_expansion, inv_key_expansion,
                 block_crypt, block_decrypt)


# =============================================================================
# Test Rijndael components
# =============================================================================

def test_mix_columns():
    """
    Test mix_columns procedure.
    Test vectors from https://en.wikipedia.org/wiki/Rijndael_MixColumns
    """
    state = np.array([[219, 19, 83, 69],
                      [242, 10, 34, 92],
                      [1, 1, 1, 1],
                      [212, 212, 212, 213]], dtype=np.uint8)
    mix_columns(state)
    fstate = np.array([[142, 77, 161, 188],
                       [159, 220, 88, 157],
                       [1, 1, 1, 1],
                       [213, 213, 215, 214]], dtype=np.uint8)
    mix_columns_success = np.all(state == fstate)
    print(f"test mix_columns: {mix_columns_success}")


def test_inverses(nb_tests: int = 10):
    """
    Test the inverse of the AES cipher components.
    """
    rng = np.random.default_rng()
    # test s_box
    s_box_success = True
    for _ in range(nb_tests):
        state = rng.integers(256, size=(4, 4), dtype=np.uint8)
        if not np.all(state == inv_s_box(s_box(state))):
            s_box_success = False
    print(f"test inv_s_box: {s_box_success}")
    # test shift_rows
    shift_rows_success = True
    for _ in range(nb_tests):
        dim = rng.integers(2, 6)
        sh = rng.integers(1, 4, size=dim)
        sh[-2:] = [4, 4]
        state = rng.integers(256, size=sh, dtype=np.uint8)
        istate = np.copy(state)
        shift_rows(state)
        inv_shift_rows(state)
        if not np.all(state == istate):
            shift_rows_success = False
    print(f"test inv_shift_rows: {shift_rows_success}")
    # test mix_columns
    mix_columns_success = True
    for _ in range(nb_tests):
        dim = rng.integers(2, 6)
        sh = rng.integers(1, 4, size=dim)
        sh[-2:] = [4, 4]
        state = rng.integers(256, size=sh, dtype=np.uint8)
        istate = np.copy(state)
        mix_columns(state)
        inv_mix_columns(state)
        if not np.all(state == istate):
            mix_columns_success = False
    print(f"test inv_mix_columns: {mix_columns_success}")


# =============================================================================
# Test key schedule-related functions
# =============================================================================

def test_key_expansion():
    """
    Test the key schedule algorithm.
    Test vector from FIPS 197-upd1: Advanced Encryption Standard (AES)
    appendix A.1.
    """
    key = bytes.fromhex("2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c")
    key = np.frombuffer(key, dtype=np.uint8)
    round_keys = key_expansion(key, 11)
    test = np.array([[0xa0, 0xfa, 0xfe, 0x17],
                     [0xca, 0xf2, 0xb8, 0xbc],
                     [0xb6, 0x63, 0x0c, 0xa6]], dtype= np.uint8)
    key_expansion_128_success = np.all(round_keys[[4, 22, 43]] == test)
    print(f"test key_expansion 128 bits: {key_expansion_128_success}")


def test_inv_key_expansion(nb_tests: int = 10):
    """
    Test key recovery from last round expanded key.
    """
    rng = np.random.default_rng(5)
    nbytes = 16
    success = True
    for _ in range(nb_tests):
        key = rng.integers(256, size=(16,), dtype=np.uint8)
        key_exp = key_expansion(key, 10)
        rkey = key_exp[-nbytes//4:].flatten()
        inv_key_exp = inv_key_expansion(rkey, 10)
        if not np.all(inv_key_exp == key_exp):
            success = False
    print(f"test inv_key_expansion: {success}")


# =============================================================================
# Test AES encryption/decrytpion
# =============================================================================

def test_block_crypt():
    """
    Test AES block cipher using a 128 bits test vector.
    Test vector from FIPS 197-upd1: Advanced Encryption Standard (AES)
    appendix B.
    """
    plain = bytes.fromhex("32 43 f6 a8 88 5a 30 8d 31 31 98 a2 e0 37 07 34")
    plain = np.copy(np.frombuffer(plain, dtype=np.uint8))
    key = bytes.fromhex("2b 7e 15 16 28 ae d2 a6 ab f7 15 88 09 cf 4f 3c")
    key = np.frombuffer(key, dtype=np.uint8)
    rkeys = key_expansion(key, 11)
    cipher = block_crypt(plain, rkeys, 10)
    test = bytes.fromhex("39 25 84 1d 02 dc 09 fb dc 11 85 97 19 6a 0b 32")
    test = np.frombuffer(test, dtype=np.uint8)
    block_cipher_success = np.all(cipher == test)
    print(f"test block_cipher 128 bits: {block_cipher_success}")


def test_block_decrypt(nb_tests: int = 10):
    """
    Test AES block decipher.
    Verify that x -> y := crypt(x) -> decrypt(y) == x
    """
    rng = np.random.default_rng(5)
    success = True
    for _ in range(nb_tests):
        plain = rng.integers(256, size=(16,), dtype=np.uint8)
        key = rng.integers(256, size=(32,), dtype=np.uint8)
        rkeys = key_expansion(key, 15)
        ciph = block_crypt(plain, rkeys, 14)
        deciph = block_decrypt(ciph, rkeys, 14)
        if not np.all(plain == deciph):
            success = False
    print(f"test block_decipher: {success}")
    

# =============================================================================
# 
# =============================================================================

if __name__ == '__main__':
    
    test_mix_columns()
    test_inverses()
    test_key_expansion()
    test_inv_key_expansion()
    
    test_block_crypt()
    test_block_decrypt()
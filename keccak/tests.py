# -*- coding: utf-8 -*-
"""
Tests for the keccak module functions.
"""

import numpy as np

from keccak import (theta, rho, pi, chi, iota,
                    inv_theta, inv_rho, inv_pi, inv_chi, inv_iota,
                    format_input, configure_hash_func,
                    keccak_round, inv_keccak_round, sha3, inv_sha3)

# =============================================================================
#Test Keccak components
# =============================================================================

def test_components(l: int = 6):
    """
    Test the different component of keccak round function and their inverse.
    """
    rng = np.random.default_rng()
    # test theta
    theta_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        if not np.all(state == inv_theta(theta(state))):
            theta_success = False
    print(f"test theta: {theta_success}")
    # test rho
    rho_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        if not np.all(state == inv_rho(rho(state))):
            rho_success = False
    print(f"test rho: {rho_success}")
    # test pi
    pi_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        if not np.all(state == inv_pi(pi(state))):
            pi_success = False
    print(f"test pi: {pi_success}")
    # test chi
    chi_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        if not np.all(state == inv_chi(chi(state))):
            chi_success = False
    print(f"test chi: {chi_success}")
    # test iota
    iota_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        i = rng.integers(0, 24)
        if not np.all(state == inv_iota(iota(state, i), i)):
            iota_success = False
    print(f"test iota: {iota_success}")


def test_round(l: int = 6):
    """
    Test the full keccak round function and its inverse.
    """
    rng = np.random.default_rng()
    # test theta
    round_success = True
    for _ in range(10):
        state = rng.binomial(1, 0.5, size=(5, 5, 2**l))
        i = rng.integers(0, 24)
        if not np.all(state == inv_keccak_round(keccak_round(state, i), i)):
            round_success = False
    print(f"test round: {round_success}")


def test_keccakf():
    """
    Test keccak_round on a state full of zeros.
    From https://github.com/Ko-/KeccakCodePackage
    """
    state = np.zeros((5, 5, 64), dtype=bool)
    for i in range(24):
        state = keccak_round(state, i)
    
    state = np.sum(state * 2**np.arange(64, dtype=np.uint64), axis=-1)
    print("Test keccak round on zeros state:",
          state[0, 0] == 0xf1258f7940e1dde7)


# =============================================================================
# ########## Test SHA3 ##########
# =============================================================================

def test_sha3():
    """
    Test SHA-3 instances with with empty message.
    Test vectors from
    https://csrc.nist.gov/projects/cryptographic-standards-and-guidelines/example-values#aHashing
    """
    data = b""
    # SHA3-224
    test = sha3(data, "SHA3-224").hex()
    control = "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7"
    print(f"test SHA3-224: {test == control}")
    # SHA3-256
    test = sha3(data, "SHA3-256").hex()
    control = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    print(f"test SHA3-256: {test == control}")
    # SHA3-384
    test = sha3(data, "SHA3-384").hex()
    control = "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2"
    control += "ac3713831264adb47fb6bd1e058d5f004"
    print(f"test SHA3-384: {test == control}")
    # SHA3-512
    test = sha3(data, "SHA3-512").hex()
    control = ("a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a"
               "615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26")
    print(f"test SHA3-512: {test == control}")
    # SHAKE128
    test = sha3(data, "SHAKE128", 256).hex()
    control = "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
    print(f"test SHAKE128: {test == control}")
    # SHAKE256
    test = sha3(data, "SHAKE256", 512).hex()
    control = ("46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762"
               "fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be")
    print(f"test SHAKE128: {test == control}")
    
    data = bytes.fromhex('00112233445566778899AABBCCDDEEFF')
    # SHAKE128
    test = sha3(data, "SHAKE128", 400).hex()
    control = ("20c2be80a64105b3481d90a6910e11f52f53af90fea76293a98730daa5c016d"
               "f1e074340ca0621faa848cf6c61793f9ba3f4")
    print(f"test SHAKE128: {test == control}")
    

def test_inv_sha3():
    """
    Test sha3 inverse.
    """
    def part_sha3(data, sha3_instance):
        sz, n, _, r, suffix = configure_hash_func(sha3_instance, None)
        state = np.zeros((sz,), dtype=bool)
        state[:r] = format_input(data, suffix, r)[0]
        state = state.reshape((5, 5, -1))
        for i in range(n):
            state = keccak_round(state, i)
        state = state.flatten()
        return np.packbits(state, bitorder='little').tobytes()
        
    data = b"this is a test"
    # SHA3-224
    s = part_sha3(data, "SHA3-224")
    test = inv_sha3(s, "SHA3-224")
    print(f"test reversed SHA3-224: {test[:len(data)] == data}")
    # SHA3-256
    s = part_sha3(data, "SHA3-256")
    test = inv_sha3(s, "SHA3-256")
    print(f"test reversed SHA3-256: {test[:len(data)] == data}")
    # SHA3-384
    s = part_sha3(data, "SHA3-384")
    test = inv_sha3(s, "SHA3-384")
    print(f"test reversed SHA3-384: {test[:len(data)] == data}")
    # SHA3-512
    s = part_sha3(data, "SHA3-512")
    test = inv_sha3(s, "SHA3-512")
    print(f"test reversed SHA3-512: {test[:len(data)] == data}")
    # SHAKE128
    s = part_sha3(data, "SHAKE128")
    test = inv_sha3(s, "SHAKE128")
    print(f"test reversed SHAKE128: {test[:len(data)] == data}")
    # SHAKE256
    s = part_sha3(data, "SHAKE256")
    test = inv_sha3(s, "SHAKE256")
    print(f"test reversed SHAKE256: {test[:len(data)] == data}")


# =============================================================================
# 
# =============================================================================

if __name__ == '__main__':
    test_components(l=6)
    test_round(l=6)
    test_keccakf()
    
    test_sha3()
    test_inv_sha3()
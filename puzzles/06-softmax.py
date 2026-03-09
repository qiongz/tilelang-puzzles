"""
Puzzle 06: Softmax
==============
Softmax is the first fundermental NN operator we learn in this tutorial.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

r"""
Softmax operator goes a little beyond the reduce sum. We also need to use serial loop to
accumulate the summation. And we need to perform an element-wise exp operation on each element
at the same time.

Note that softmax needs to be computed in numerically stable form as in Python. To achieve this,
we need to subtract the maximum value of each row from all elements in that row
before applying the exponential function.

HINT:
1. Use `T.fill` to set the initial value of the buffer. `T.clear` sets all elements to zero by
default, which may not be what you want.

3.We recommend not using `T.exp` but instead using `T.exp2`. You need the identity

.. math::
    \exp(x) = 2^{\log_2(e) x}

The constant log2_e is provided.

BONUS: Use "Online Softmax" algorithm to implement optimized softmax. This is also a core idea of
FlashAttention algorithm. Through this, we can implement softmax with only two passes / loops.

06-1: Softmax.

Inputs:
    A: Tensor([N, M], float32)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384

Output:
    B: Tensor([N, M], float16)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row

Definition:
    for i in range(N):
        S = 0
        MAX = -inf
        for j in range(M):
            MAX = max(A[i, j], MAX)
        for j in range(M):
            B[i, j] = exp(A[i, j] - MAX)
            SUM += B[i, j]
        for j in range(M):
            B[i, j] /= SUM
"""


def ref_softmax(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.softmax(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_softmax(A, BLOCK_N: int, BLOCK_M: int):
    """ Two-Pass Softmax algorithm """
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N, M), dtype)

    # TODO: Implement this function
    grid_x = T.ceildiv(N , BLOCK_N)
    grid_y = T.ceildiv(M , BLOCK_M)

    with T.Kernel(grid_x, threads = 256) as bx:
        A_local = T.alloc_fragment((BLOCK_N, BLOCK_M),dtype)
        out_local = T.alloc_fragment((BLOCK_N, BLOCK_N), dtype)

        max_tmp = T.alloc_fragment(BLOCK_N, dtype)
        sum_tmp = T.alloc_fragment(BLOCK_N, dtype)

        # reduce max/sum rowwise
        max_local = T.alloc_fragment(BLOCK_N, dtype)
        sum_local = T.alloc_fragment(BLOCK_N, dtype)
        T.fill(max_local, -1e30)
        T.clear(sum_local)


        for by in T.serial(grid_y):
            T.copy(A[bx * BLOCK_N, by * BLOCK_M], A_local)
            # max_tmp: block-rowwise reduced max
            T.reduce_max(A_local, max_tmp, dim=1, clear=True)

            # block-rowwise-max
            # max_tmp: new-maximum within this block
            for i in T.Parallel(BLOCK_N):
                max_tmp[i] = T.max(max_local[i], max_tmp[i])

            # exp(x_i - m_i)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                out_local[i, j] = T.exp2(log2_e * (A_local[i, j] - max_local[i]))

            # block-rowwise-sum
            T.reduce_sum(out_local, sum_tmp, dim=1, clear=True)

            # max_local:  m_i <- max(m_{i-1}, x_i)
            # sum_local:  sum_i <- sum_{i-1} * exp(m_{i-1} - m_i) + exp(x_i - m_i)
            for i in T.Parallel(BLOCK_N):
                m_new = max_tmp[i]
                m_old = max_local[i]
                alpha = T.exp2(log2_e * (m_old - m_new))
                sum_local[i] = sum_local[i] * alpha + sum_tmp[i]
                max_local[i] = m_new

        for by in T.serial(grid_y):
            T.copy(A[bx * BLOCK_N, by * BLOCK_M], A_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                out_local[i, j] = T.exp2(log2_e * (A_local[i, j] - max_local[i])) / sum_local[i]

            T.copy(out_local, B[bx * BLOCK_N, by * BLOCK_M])

    return B


def run_softmax():
    print("\n=== Softmax ===\n")
    N = 4096
    M = 16384
    BLOCK_N = 16
    BLOCK_M = 256
    test_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
    )
    bench_puzzle(
        tl_softmax,
        ref_softmax,
        {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_softmax()

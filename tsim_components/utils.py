import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set

# handle jagged ndarray
def pad_to_dense(M:List[List[int]]) -> np.ndarray:
    maxlen = max(len(r) for r in M)
    Z = np.zeros((len(M), maxlen),dtype=int)-1  # type: ignore
    for enu, row in enumerate(M):
        Z[enu, :len(row)] = row
    return Z

# convert per var variables to per var shapes
def var_to_shape(dim_lengths_np:np.ndarray,
                 variable:np.ndarray) -> np.ndarray:
    return dim_lengths_np[variable]    

# convert per var shapes to per var sizes (# elements)
def shape_to_size(shape:np.ndarray) -> int:
    if np.shape(shape)[1]>1:    # e.g., 1*1 kernel will not cause additional input size
        shape[:,1:][shape[:,1:]>0] -= 1
    return int(np.prod(np.sum(shape,axis=-1)))

def _variable_dimension_set(variable: List[List[int]]) -> Set[int]:
    """Return the iteration-space dimensions referenced by one tensor."""
    return {
        int(index) for term in variable for index in term if int(index) >= 0
    }


def matmul_dimension_groups(
    variables: List[List[List[int]]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Infer flattened ``(B, K, M, N)`` dimension groups for an einsum.

    VOXEL represents a matmul as three tensor-variable maps in output, input
    A, input B order.  Transformer projections frequently flatten more than
    one logical axis into M, N, or K (for example ``M=B*S`` and
    ``K=heads*head_dim``), so inspecting only the final two axes is not
    sufficient.

    Returns four disjoint tuples of iteration-space dimension indices:

    * B: present in output and both inputs;
    * K: present in both inputs but absent from output;
    * M: present in output and input A but absent from input B;
    * N: present in output and input B but absent from input A.
    """
    if len(variables) != 3:
        raise ValueError(
            f"matmul requires output and two inputs, got {len(variables)} tensors"
        )
    output = _variable_dimension_set(variables[0])
    input_a = _variable_dimension_set(variables[1])
    input_b = _variable_dimension_set(variables[2])

    batch = output & input_a & input_b
    contraction = (input_a & input_b) - output
    m_dims = (output & input_a) - input_b
    n_dims = (output & input_b) - input_a

    if not contraction:
        raise ValueError(
            "matmul has no contraction dimension shared by both inputs and "
            f"absent from the output: output={sorted(output)}, "
            f"input_a={sorted(input_a)}, input_b={sorted(input_b)}"
        )
    represented_output = batch | m_dims | n_dims
    if represented_output != output:
        raise ValueError(
            "matmul output contains dimensions that cannot be assigned to "
            f"B/M/N: {sorted(output - represented_output)}"
        )
    represented_a = batch | m_dims | contraction
    represented_b = batch | n_dims | contraction
    if represented_a != input_a or represented_b != input_b:
        raise ValueError(
            "matmul inputs contain unsupported broadcast/reduction dimensions: "
            f"input_a_extra={sorted(input_a - represented_a)}, "
            f"input_b_extra={sorted(input_b - represented_b)}"
        )

    return tuple(sorted(batch)), tuple(sorted(contraction)), \
        tuple(sorted(m_dims)), tuple(sorted(n_dims))


def _dimension_product(dim_lengths: List[int], indices: Tuple[int, ...]) -> int:
    return int(np.prod([dim_lengths[index] for index in indices], dtype=np.int64)) \
        if indices else 1


# convert arbitrary einsum dimension groups to flattened b,k,m,n for matmul
def dim_var_to_bkmn(dim_lengths:List[int],
                    variables:List[List[List[int]]]) -> Tuple[int,int,int,int]:
    batch, contraction, m_dims, n_dims = matmul_dimension_groups(variables)
    return (
        _dimension_product(dim_lengths, batch),
        _dimension_product(dim_lengths, contraction),
        _dimension_product(dim_lengths, m_dims),
        _dimension_product(dim_lengths, n_dims),
    )

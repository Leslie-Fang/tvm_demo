import tvm
from tvm import te
from tvm.te.tensor import Tensor
from tvm.te.schedule import Schedule
from tvm.runtime.module import Module
import timeit

def vector_add(n):
    A = te.placeholder((n,), dtype="float", name="a")
    B = te.placeholder((n,), dtype="float", name="b")
    C = te.compute((n,), lambda i: A[i] + B[i], name='c')

    #create the default schedule
    s = te.create_schedule(C.op)

    return s, (A,B,C)

def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return A, B, C

if __name__ == "__main__":

    n = 100
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target='llvm')

    print("----------------")
    #print(mod)
    print(mod.get_source('ll'))
    print("---------------2")
    print(mod.entry_func)
    print("---------------3")
    print(mod.type_key)
    print("---------------4")
    print(mod.type_key)

    # Run the test
    print("---------------Start execute")
    import numpy as np
    ctx = tvm.context('llvm', 0)

    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n, n)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros((n, n), dtype=C.dtype), ctx)
    mod(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy()), rtol=0.001, atol=0.001)

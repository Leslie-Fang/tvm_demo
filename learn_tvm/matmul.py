import tvm
from tvm import te
from tvm.te.tensor import Tensor
from tvm.te.schedule import Schedule
from tvm.runtime.module import Module
import timeit
import numpy as np
import os
import argparse
from tvm import autotvm

num_threads = 28
os.environ["TVM_NUM_THREADS"] = str(num_threads)

target = 'llvm -mcpu=skylake-avx512'
ctx = tvm.cpu()

def bench_workload(workload):
    """Benchmark a workload

    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

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

def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def default(n):
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)
    #print(tvm.lower(s, [A, B], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def reorder(n):
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)

    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)

    #print(tvm.lower(s, [A, B], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def vectorize(n):
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)

    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    s[C].vectorize(y)

    print(tvm.lower(s, [A, B, C], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def parallel(n):
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)

    (x, y), (k,) = C.op.axis, C.op.reduce_axis
    s[C].reorder(x, k, y)
    s[C].vectorize(y)
    s[C].parallel(C.op.axis[0])

    print(tvm.lower(s, [A, B], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def block(n):
    tx, ty, tk = 64, 64, 16  #tile sizes
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)
    # Tile by blocks, and then parallelize the computation of each block
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # Optimize the computation of each block
    # k = s[C].op.reduce_axis[0]
    ko, ki = s[C].split(s[C].op.reduce_axis[0], factor=tk)
    #s[C].reorder(xi, ko, ki, yi)
    s[C].reorder(ko, xi, ki, yi)
    s[C].unroll(ki)
    s[C].vectorize(yi)

    print(tvm.lower(s, [A, B], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def cache(n):
    tx, ty, tk = 64, 64, 16  # tile sizes
    A, B, C = matmul(n, n, n)
    s = te.create_schedule(C.op)

    # Create a write cache for C
    CachedC = s.cache_write(C, 'local')
    # Same as before, first tile by blocks, and then parallelize the
    # computation of each block
    xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
    xy = s[C].fuse(xo, yo)
    s[C].parallel(xy)
    # Use the write cache for the output of the xy axis, namely a block.
    s[CachedC].compute_at(s[C], xy)
    # Same as before to optimize the computation of a block .
    xc, yc = s[CachedC].op.axis
    ko, ki = s[CachedC].split(CachedC.op.reduce_axis[0], factor=tk)
    s[CachedC].reorder(ko, xc, ki, yc)
    s[CachedC].unroll(ki)
    s[CachedC].vectorize(yc)

    print(tvm.lower(s, [A, B], simple_mode=True))
    mod = tvm.build(s, [A, B, C], target=target)
    return mod

def autoTVM_matmul(n):

    @autotvm.template("leslie/matmul")
    def my_tune(n):
        #tx, ty, tk = 64, 64, 16  # tile sizes

        cfg = autotvm.get_config()
        cfg.define_knob("tile_x", [16, 64, 128])
        cfg.define_knob("tile_y", [16, 64, 128])
        cfg.define_knob("tile_k", [4, 16, 64, 128])

        tx = cfg["tile_x"].val
        ty = cfg["tile_y"].val
        tk = cfg["tile_k"].val

        A, B, C = matmul(n, n, n)
        s = te.create_schedule(C.op)

        # Create a write cache for C
        CachedC = s.cache_write(C, 'local')
        # Same as before, first tile by blocks, and then parallelize the
        # computation of each block
        xo, yo, xi, yi = s[C].tile(*C.op.axis, tx, ty)
        xy = s[C].fuse(xo, yo)
        s[C].parallel(xy)
        # Use the write cache for the output of the xy axis, namely a block.
        s[CachedC].compute_at(s[C], xy)
        # Same as before to optimize the computation of a block .
        xc, yc = s[CachedC].op.axis
        ko, ki = s[CachedC].split(CachedC.op.reduce_axis[0], factor=tk)
        s[CachedC].reorder(ko, xc, ki, yc)
        s[CachedC].unroll(ki)
        s[CachedC].vectorize(yc)

        return s, [A, B, C]

    task = autotvm.task.create("leslie/matmul", args=[n], target=target)
    print(task.config_space)
    print(len(task.config_space))
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(timeout=10000),
                                            runner=autotvm.LocalRunner(repeat=1, number=10, min_repeat_ms=1000, timeout=1000))

    logfile = 'leslie_tune.log'
    os.system("rm -rf {}".format(logfile))
    tuner = autotvm.tuner.XGBTuner(task)
    n_trial = len(task.config_space)
    prefix = "[Task]"
    tuner.tune(n_trial=n_trial, measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix),
                                                                          autotvm.callback.log_to_file(logfile)])

    # evalute task
    with autotvm.apply_history_best(logfile):
        print("Compiling")
        with tvm.target.create(target):
            s, arg_bufs = my_tune(n)

            print(tvm.lower(s, arg_bufs, simple_mode=True))

            mod = tvm.build(s, arg_bufs, target=target)
    return mod


if __name__ == "__main__":
    n = 1024
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()

    print(args.save)
    save = args.save
    path_o = "./export/deploy_lib.o"
    path_lib = "./export/deploy_lib.so"
    if save:
        #mod = default(n)
        #mod = reorder(n)
        #mod = vectorize(n)
        #mod = parallel(n)
        mod = block(n)
        #mod = cache(n)
        #mod = autoTVM_matmul(n)

        os.system("rm -rf ./export && mkdir export")
        mod.export_library(path_lib)

        from tvm.contrib import cc
        mod.save(path_o)
        cc.create_shared(path_lib, [path_o])
    else:
        #Benchmark
        mod = tvm.runtime.load_module(path_lib)
        # assert mod.type_key == "library"
        # assert mod.imported_modules[0].type_key == "cuda"

        a, b, c = get_abc((n, n), lambda x: tvm.nd.array(x, ctx=ctx))

        def workload(nrepeats):
            timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
            return timer(a, b, c).mean * nrepeats

        res = bench_workload(workload)
        print("runtime: {} s".format(res))
        gflops = 2 * n * n**2 / 1e9 / res # res is time, 2*n (each line has n mul and n add), total n*n vector operation
        print("gflops: {}".format(gflops))

        # Compare the result with numpy
        outnp = c.asnumpy()
        t_out = np.matmul(a.asnumpy(), b.asnumpy())
        np.testing.assert_allclose(t_out, outnp, atol=1e-3)





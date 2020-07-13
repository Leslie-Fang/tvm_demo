import tvm
from tvm import te
from tvm.te.tensor import Tensor
from tvm.te.schedule import Schedule
from tvm.runtime.module import Module
import timeit
import numpy as np
import os
import argparse

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

def get_conv_data(bs, oc, ic, n, k, p=0, s=1, constructor=None):
    """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output
    tensor with the shapes specified by input arguments.

    bs: batchsize
    oc, ic : output and input channels
    n : input width and height
    k : kernel width and height
    p : padding size, default 0
    s : stride, default 1
    constructor : user-defined tensor constructor
    """
    np.random.seed(0)
    data = np.random.normal(size=(bs, ic, n, n)).astype('float32')
    weight = np.random.normal(size=(oc, ic, k, k)).astype('float32')
    on = conv_out_size(n, k, p, s)
    out = np.empty((bs, oc, on, on), dtype='float32')
    if constructor:
        data, weight, out = (constructor(x) for x in [data, weight, out])
    return data, weight, out

def conv_out_size(n, k, p, s):
    """Compute the output size by given input size n (width or height),
    kernel size k, padding p, and stride s
    Return output size (width or height)
    """
    return (n - k + 2 * p)//s + 1

def padding(X, ph, pw, val=0):
    """Pad X with the given value in 2-D

    ph, pw : height and width padding
    val : padding value, default 0
    """
    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute(
            (*X.shape[0:-2], nh+ph*2, nw+pw*2),
            lambda *i: te.if_then_else(
                te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
                val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
            name='PaddedX')


def conv_gflop(bs, oc, ic, n, k, p, s):
    """Compute the #floating point operations in a convolution.

    The arguments are output channels oc, input channels ic, input size n,
    kernel size k, padding p and stride s.
    """
    on = conv_out_size(n, k, p, s)
    return 2 * bs * oc * ic * on * on * k * k / 1e9

def default_conv(bs, oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((bs, ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (bs, oc, oh, ow),
        lambda b, c, i, j: te.sum(
            PaddedX[b, ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
            axis=[ric, rkh, rkw]), name='Y')

    sch = te.create_schedule(Y.op)

    # # parallel calculate each element in the output
    # bs, oc, h, w = Y.op.axis
    # tile_w = 16
    # wo, wi = sch[Y].split(w, tile_w)
    #
    # bsoc = sch[Y].fuse(bs, oc, h, wo)
    # sch[Y].parallel(bsoc)
    # sch[Y].vectorize(wi)

    #rickhkw = sch[Y].fuse(ric, rkh, rkw)


    print(tvm.lower(sch, [X, K, Y], simple_mode=True))

    mod = tvm.build(sch, [X, K, Y], target=target)

    return mod

def cached_block_conv(bs, oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """Convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    ric = te.reduce_axis((0, ic), name='ric')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((bs, ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    Y = te.compute(
        (bs, oc, oh, ow),
        lambda b, c, i, j: te.sum(
            PaddedX[b, ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
            axis=[ric, rkh, rkw]), name='Y')

    sch = te.create_schedule(Y.op)

    CachedY = sch.cache_write(Y, 'local')
    # Compute the output block for every output channel in parallel
    bs, oc, h, w = Y.op.axis
    th, tw = 8, 8 # Tile sizes for height and weight
    ho, wo, hi, wi = sch[Y].tile(h, w, th, tw)
    ochw = sch[Y].fuse(oc, ho, wo)
    sch[Y].parallel(ochw)

    # Cache the output block, and move the inner height and width axes
    # to innermost, so we can vectorize and unroll them
    sch[CachedY].compute_at(sch[Y], ochw)
    _, _,  ch, cw = CachedY.op.axis
    ric, rkh, rkw = CachedY.op.reduce_axis
    sch[CachedY].reorder(ric, rkh, rkw, ch, cw)
    sch[CachedY].vectorize(cw)
    sch[CachedY].unroll(ch)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        sch[PaddedX].parallel(PaddedX.op.axis[0])

    print(tvm.lower(sch, [X, K, Y], simple_mode=True))

    mod = tvm.build(sch, [X, K, Y], target=target)

    return mod

def packed_conv(bs, oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2-D conv

    oc, ic : output and input channels.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    toc, tic : the tiling sizes of output channel and input channel
    """
    """Pack data and weight for convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    toc, tic : the tiling sizes of the output and input channels
    """
    toc = 16
    tic = 16
    tw = 4
    X = te.placeholder((bs, ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    # pack X and K
    assert ic % tic == 0 and oc % toc == 0
    PackedX = te.compute(
        (bs, ic//tic, nh+ph*2, nw+pw*2, tic),
        lambda b, ic_out, x, y, ic_in: PaddedX[b, ic_out*tic + ic_in, x, y],
        name='PackedX')
    PackedK = te.compute(
        (oc//toc, ic//tic, kh, kw, tic, toc),
        lambda oc_out, ic_out, x, y, ic_in, oc_in: K[
            oc_out*toc + oc_in, ic_out*tic + ic_in, x, y],
        name='PackedK')

    # reduction axes
    ric_in = te.reduce_axis((0, tic), name='ric_in')
    ric_out = te.reduce_axis((0, ic//tic), name='ric_out')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # Compuated Y in the packed layout
    PackedY = te.compute(
        (bs, oc//toc, oh, ow, toc),
        lambda b, oc_out, x, y, oc_in: te.sum(
            PackedX[b, ric_out, x*sh+rkh, y*sw+rkw, ric_in] *
            PackedK[oc_out, ric_out, rkh, rkw, ric_in, oc_in],
            axis=[ric_out, rkh, rkw, ric_in]), name='Y')

    # Unpack the result
    Y = te.compute((bs, oc, oh, ow),
                    lambda b, oc, x, y: PackedY[b, oc//toc, x, y, oc%toc],
                    name='Y')

    s = te.create_schedule(Y.op)
    CachedY = s.cache_write(PackedY, 'local')
    bso, oc_out, h, w, oc_in = s[PackedY].op.axis
    oc_out_h = s[PackedY].fuse(bso, oc_out, h)
    # Parallel on the first two dimensions oc_out and h
    s[PackedY].parallel(oc_out_h)
    # Optimize the computation of a cached output block
    w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns

    s[CachedY].compute_at(s[PackedY], w_out)

    _, _, _, cw, oc_in = CachedY.op.axis
    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis

    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    s[CachedY].unroll(ric_in)
    s[CachedY].unroll(cw)
    s[CachedY].vectorize(oc_in)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    s[PackedX].parallel(s[PackedX].fuse(*PackedX.op.axis[0:2]))
    s[PackedX].unroll(PackedX.op.axis[-1])
    s[PackedK].parallel(s[PackedK].fuse(*PackedK.op.axis[0:2]))
    s[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    s[Y].parallel(s[Y].fuse(*Y.op.axis[0:2]))
    s[Y].unroll(Y.op.axis[-1])

    print(tvm.lower(s, [X, K, Y], simple_mode=True))

    mod = tvm.build(s, [X, K, Y], target=target)

    return mod

def test_packed_conv(bs, oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2-D conv

    oc, ic : output and input channels.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    toc, tic : the tiling sizes of output channel and input channel
    """
    """Pack data and weight for convolution

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    toc, tic : the tiling sizes of the output and input channels
    """
    toc = 16
    tic = 16
    tw = 4
    X = te.placeholder((bs, ic, nh, nw), name='X')
    K = te.placeholder((oc, ic, kh, kw), name='K')
    PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
    # pack X and K
    assert ic % tic == 0 and oc % toc == 0

    PackedX = te.compute(
        (bs, ic//tic, nh+ph*2, nw+pw*2, tic),
        lambda b, ic_out, x, y, ic_in: PaddedX[b, ic_out*tic + ic_in, x, y],
        name='PackedX')

    PackedK = te.compute(
        (oc//toc, ic//tic, kh, kw, tic, toc),
        lambda oc_out, ic_out, x, y, ic_in, oc_in: K[
            oc_out*toc + oc_in, ic_out*tic + ic_in, x, y],
        name='PackedK')

    # reduction axes
    ric_in = te.reduce_axis((0, tic), name='ric_in')
    ric_out = te.reduce_axis((0, ic//tic), name='ric_out')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)

    # Compuated Y in the packed layout
    PackedY = te.compute(
        (bs, oc//toc, oh, ow, toc),
        lambda b, oc_out, x, y, oc_in: te.sum(
            PackedX[b, ric_out, x*sh+rkh, y*sw+rkw, ric_in] *
            PackedK[oc_out, ric_out, rkh, rkw, ric_in, oc_in],
            axis=[ric_out, rkh, rkw, ric_in]), name='Y')

    # Unpack the result
    Y = te.compute((bs, oc, oh, ow),
                    lambda b, oc, x, y: PackedY[b, oc//toc, x, y, oc%toc],
                    name='Y')

    s = te.create_schedule(Y.op)

    CachedY = s.cache_write(PackedY, 'local')
    bso, oc_out, h, w, oc_in = s[PackedY].op.axis

    # self test by leslie
    s[PackedY].reorder(bso, h, w, oc_out, oc_in)

    #bso, h, w, oc_out, oc_in = s[PackedY].op.axis

    w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns

    bso_h_w_out = s[PackedY].fuse(bso, h, w_out)
    s[PackedY].parallel(bso_h_w_out)

    s[CachedY].compute_at(s[PackedY], bso_h_w_out)

    _, _, cw, c_oc_out, c_oc_in = CachedY.op.axis

    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis

    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, c_oc_out, c_oc_in)

    s[CachedY].unroll(cw)
    s[CachedY].unroll(c_oc_out)
    s[CachedY].vectorize(c_oc_in)

    # _, _, cw, c_oc_out, c_oc_in = PackedY.op.axis
    # ric_out, rkh, rkw, ric_in = PackedY.op.reduce_axis

    # s[PackedY].reorder(ric_out, rkh, rkw, ric_in, cw, c_oc_out, c_oc_in)
    # s[PackedY].unroll(cw)
    # s[PackedY].unroll(c_oc_out)
    # s[PackedY].vectorize(c_oc_in)



    # oc_out_h = s[PackedY].fuse(bso, oc_out, h)
    # # Parallel on the first two dimensions oc_out and h
    # s[PackedY].parallel(oc_out_h)
    # # Optimize the computation of a cached output block
    # w_out, w_in = s[PackedY].split(w, factor=tw)  # Split the columns
    #
    # s[CachedY].compute_at(s[PackedY], w_out)
    #
    # _, _, _, cw, oc_in = CachedY.op.axis
    # ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis
    #
    # s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    # s[CachedY].unroll(ric_in)
    # s[CachedY].unroll(cw)
    # s[CachedY].vectorize(oc_in)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    s[PackedX].parallel(s[PackedX].fuse(*PackedX.op.axis[0:2]))
    s[PackedX].unroll(PackedX.op.axis[-1])
    s[PackedK].parallel(s[PackedK].fuse(*PackedK.op.axis[0:2]))
    s[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    s[Y].parallel(s[Y].fuse(*Y.op.axis[0:2]))
    s[Y].unroll(Y.op.axis[-1])

    print(tvm.lower(s, [X, K, Y], simple_mode=True))

    mod = tvm.build(s, [X, K, Y], target=target)

    return mod

if __name__ == "__main__":
    #n = 2048
    bs = 128
    oc, ic, n, k, p, s = 64, 64, 56, 3, 1, 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()

    print(args.save)
    save = args.save
    path_o = "./export/deploy_lib.o"
    path_lib = "./export/deploy_lib.so"
    if save:
        #mod = default_conv(bs, oc, ic, n, n, k, k, p, p, s, s)
        #mod = cached_block_conv(bs, oc, ic, n, n, k, k, p, p, s, s)
        #mod = packed_conv(bs, oc, ic, n, n, k, k, p, p, s, s)
        mod = test_packed_conv(bs, oc, ic, n, n, k, k, p, p, s, s)

        os.system("rm -rf ./export && mkdir export")
        mod.export_library(path_lib)

        from tvm.contrib import cc
        mod.save(path_o)
        cc.create_shared(path_lib, [path_o])
    else:
        #Benchmark
        mod = tvm.runtime.load_module(path_lib)
        data, weight, out = get_conv_data(bs, oc, ic, n, k, p, s, tvm.nd.array)

        def workload(nrepeats):
            timer = mod.time_evaluator(mod.entry_name, ctx=ctx, number=nrepeats)
            return timer(data, weight, out).mean * nrepeats

        res = bench_workload(workload)
        print("runtime: {} s".format(res))

        gflops = conv_gflop(bs, oc, ic, n, k, p, s) / res
        print("gflops: {}".format(gflops))

        outnp = out.asnumpy()
        print(type(outnp))
        print(outnp.shape)


        import tensorflow as tf
        print(type(data))
        print(data.asnumpy().shape)
        print(weight.asnumpy().shape)
        data = np.transpose(data.asnumpy(), (0, 2, 3, 1))
        weight = np.transpose(weight.asnumpy(), (2, 3, 1, 0))
        t_out = tf.nn.conv2d(data, weight, strides=[1, 1, 1, 1], padding='SAME')
        t_out = t_out.numpy()
        t_out = np.transpose(t_out, (0, 3, 1, 2))
        print(type(t_out))

        print("-----------------")
        print(t_out.shape)
        print(outnp.shape)
        np.testing.assert_allclose(t_out, outnp, atol=1e-3)






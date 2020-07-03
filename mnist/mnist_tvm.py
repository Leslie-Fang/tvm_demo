# tvm, relay
import tvm
from tvm import te
from tvm import relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.python.framework import dtypes

target = 'llvm'
# target_host = 'llvm'
target_host = None
layout = None
# ctx = tvm.cpu(0)


model_path = "/home/lesliefang/tvm/test_script/mnist/pb_models/freeze_fp32.pb"
INPUTS = 'X'
OUTPUTS = 'Ys'

# model_path = "/home/lesliefang/tvm/test_script/mobilenet_v1/mobilenet_v1_1.0_224_frozen-with-shapes.pb"
# model_path = "/root/.tvm_test_data/tf/InceptionV1/classify_image_graph_def-with_shapes.pb"

with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())

    graph = tf_compat_v1.import_graph_def(graph_def, name='')

    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, OUTPUTS)
        #graph_def = tf_testing.AddShapesToGraphDef(sess, 'MobilenetV1/Predictions/Reshape_1')
        #graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

shape_dict = {INPUTS: (1, 784)}
#shape_dict = None

mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")

with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)

# save the graph, lib and params into separate files
from tvm.contrib import util

# temp = util.tempdir("./save")
path_lib = "./export/deploy_lib.tar"
lib.export_library(path_lib)
with open("./export/deploy_graph.json", "w") as fo:
    fo.write(graph)
with open("./export/deploy_param.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

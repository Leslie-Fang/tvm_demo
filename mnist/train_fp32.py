# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
import time
import os
from dataset_help import train_images
from dataset_help import train_labels
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import node_def_pb2, graph_pb2, attr_value_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    log_step = 250
    learning_rate = 0.001
    batchsize = 128
    epoch = 1

    base_path = os.path.join(os.getcwd(), "dataset")
    train_image_path = os.path.join(base_path, "train-images-idx3-ubyte")
    train_label_path = os.path.join(base_path, "train-labels-idx1-ubyte")
    train_labels_data = train_labels(train_label_path)
    train_images_data = train_images(train_image_path)

    input_image_size = int(train_images_data.get_row_number()) * int(train_images_data.get_column_number())

    X = tf.compat.v1.placeholder(tf.float32, [None, input_image_size], name="X")
    x_image = tf.reshape(X, [-1, 28, 28, 1])

    Y_ = tf.compat.v1.placeholder(tf.float32, [None, 10], name="Y_")  # 10表示手写数字识别的10个类别

    # conv1
    layer1_weights = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([5, 5, 1, 32], stddev=0.05))
    layer1_bias = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[32]))
    layer1_conv = tf.compat.v1.nn.conv2d(x_image, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')  # https://www.cnblogs.com/lizheng114/p/7498328.html
    layer1_relu = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(layer1_conv, layer1_bias))  # https://blog.csdn.net/m0_37870649/article/details/80963053
    layer1_pool = tf.compat.v1.nn.max_pool(layer1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv2
    layer2_weights = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([5, 5, 32, 64], stddev=0.05))
    layer2_bias = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[64]))
    layer2_conv = tf.compat.v1.nn.conv2d(layer1_pool, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(layer2_conv, layer2_bias))
    layer2_pool = tf.compat.v1.nn.max_pool(layer2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC
    layer3_weights = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([7 * 7 * 64, 1024], stddev=0.05))
    layer3_bias = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[1024]))
    layer3_flat = tf.compat.v1.reshape(layer2_pool, [-1, 7 * 7 * 64])  # 展开成一维，进行全连接层的计算
    layer3_relu = tf.compat.v1.nn.relu(tf.compat.v1.nn.bias_add(tf.matmul(layer3_flat, layer3_weights), layer3_bias))

    # FC2
    layer4_weights = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([1024, 10], stddev=0.05))
    layer4_bias = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[10]))
    layer4_optput = tf.compat.v1.nn.bias_add(tf.compat.v1.matmul(layer3_relu, layer4_weights), layer4_bias)

    Ys = tf.nn.softmax(layer4_optput, name="Ys")

    loss = -tf.reduce_sum(Y_ * tf.compat.v1.log(Ys))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    total_time_ms = 0
    global_step = 0
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=28,
                                      inter_op_parallelism_threads=1)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Begin train!")

        for e in range(epoch):
            if train_labels_data is not None:
                del train_labels_data
                train_labels_data = train_labels(train_label_path)
            if train_images_data is not None:
                del train_images_data
                train_images_data = train_images(train_image_path)
            print("train_images_data.get_images_number() is:{}".format(train_images_data.get_images_number()))
            iterations = train_images_data.get_images_number() / batchsize
            for step in range(int(iterations)):
                # label_val = train_labels_data.read_one_label()
                # train_image_pixs = train_images.read_one_image("{0}/{1}_label_{2}.png".format("/home/mnist_dataset/train_data/images",step+1,label_vals[0]))
                label_vals = train_labels_data.read_labels(batchsize)
                train_image_pixs = train_images_data.read_images(batchsize)
                train_y_label = []
                for item in label_vals:
                    train_sub_y_label = []
                    for i in range(10):
                        if item != i:
                            train_sub_y_label.append(0)
                        else:
                            train_sub_y_label.append(1)
                    train_y_label.append(train_sub_y_label)
                train_x = np.array(train_image_pixs, dtype=np.float32)
                train_y = np.array(train_y_label, dtype=np.float32)
                start_time = time.time()
                if global_step == 300:
                    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                    run_metadata = tf.compat.v1.RunMetadata()
                    sess.run(train_op, feed_dict={X: train_x, Y_: train_y}, options=run_options, run_metadata=run_metadata)
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('fp32_train_timeline.json', 'w') as f:
                        f.write(ctf)
                else:
                    sess.run(train_op, feed_dict={X: train_x, Y_: train_y})
                end_time = time.time()
                step_time_ms = (end_time-start_time) * 1000
                total_time_ms += step_time_ms
                global_step += 1

                if (int(step) % int(log_step)) == 0:
                    c = sess.run(loss, feed_dict={X: train_x, Y_: train_y})
                    print("epoch:{0}, Step:{1}, loss:{2}, time in ms:{3}".format(e + 1, step, c, step_time_ms))
        print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate, batchsize, epoch))
        print("Global step is:{}".format(global_step))
        print("The program takes:{0} msec, aveagre step time: {1} msec, average thoughput is: {2} fps".
              format(total_time_ms, float(total_time_ms)/global_step, batchsize*global_step*1000/float(total_time_ms)))
        if os.path.isdir(os.path.join(os.getcwd(), "checkPoint")) is False:
            os.makedirs(os.path.join(os.getcwd(), "checkPoint"))
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, os.path.join(os.getcwd(), "checkPoint/trainModel"))

        if os.path.isdir(os.path.join(os.getcwd(), "pb_models")) is False:
            os.makedirs(os.path.join(os.getcwd(), "pb_models"))
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Ys'])
        with tf.compat.v1.gfile.FastGFile(os.path.join(os.getcwd(), "pb_models") + '/freeze_fp32.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

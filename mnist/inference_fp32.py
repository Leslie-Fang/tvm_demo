# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
import time
import os
from dataset_help import inference_images
from dataset_help import inference_labels
from tensorflow.python.framework import graph_util
from tensorflow.core.framework import node_def_pb2, graph_pb2, attr_value_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
	print("Begin inference!")
	#base_path = "/home/mnist_dataset"
	base_path = os.getcwd()
	base_inference_path = os.path.join(base_path,"dataset")
	inference_image_path = os.path.join(base_inference_path,"t10k-images-idx3-ubyte")
	inference_label_path = os.path.join(base_inference_path,"t10k-labels-idx1-ubyte")
	inference_labels = inference_labels(inference_label_path)
	inference_images = inference_images(inference_image_path)
	input_image_size = int(inference_images.get_row_number())*int(inference_images.get_column_number())
	right_count = 0
	batchsize = 28
	total_time_ms = 0
	global_step = 0

	config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=28,
							inter_op_parallelism_threads=1)

	with tf.compat.v1.Session(config=config) as sess:
		# saver = tf.train.import_meta_graph(os.path.join(base_path,"train_data/checkPoint/trainModel.meta"))
		# saver.restore(sess, tf.train.latest_checkpoint(os.path.join(base_path,"train_data/checkPoint")))

		with gfile.FastGFile(os.path.join(base_path, "pb_models") + '/freeze_fp32.pb', 'rb') as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())
			for node in graph_def.node:
				print("node name is: {} \t node op is: {}".format(node.name, node.op))
			sess.graph.as_default()
			tf.compat.v1.import_graph_def(graph_def, name='')  # 导入计算图

		iterations = inference_images.get_images_number()/batchsize
		for step in range(int(iterations)):
			label_vals = inference_labels.read_labels(batchsize)
			inference_image_pixs = inference_images.read_images(batchsize)

			#print(label_vals)
			inference_y_label = []
			for item in label_vals:
				inference_sub_y_label = []
				for i in range(10):
					if item != i:
						inference_sub_y_label.append(0)
					else:
						inference_sub_y_label.append(1)
				inference_y_label.append(inference_sub_y_label)
			inference_x = np.array(inference_image_pixs, dtype=np.float32)
			inference_y = np.array(inference_y_label, dtype=np.float32)
			#print(inference_y.shape)
			# 获取需要进行计算的operator
			Ys = sess.graph.get_tensor_by_name('Ys:0')
			X = sess.graph.get_tensor_by_name('X:0')
			# Y_ = sess.graph.get_tensor_by_name('Y_:0')
			start_time = time.time()
			if global_step == 300:
				run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
				run_metadata = tf.compat.v1.RunMetadata()
				results = sess.run(Ys, feed_dict={X: inference_x}, options=run_options, run_metadata=run_metadata)
				tl = timeline.Timeline(run_metadata.step_stats)
				ctf = tl.generate_chrome_trace_format()
				with open('fp32_inference_timeline.json', 'w') as f:
					f.write(ctf)
			else:
				results = sess.run(Ys,feed_dict={X:inference_x})
			end_time = time.time()
			step_time_ms = (end_time - start_time) * 1000
			total_time_ms += step_time_ms
			global_step += 1

			for image_number in range(batchsize):
				maxindex  = np.argmax(results[image_number])
				true_label = np.argmax(inference_y[image_number])
				# print(results.shape)
				# print(inference_y.shape)
				if maxindex == true_label:
					right_count = right_count + 1

		print("running batchsize is:{}".format(batchsize))
		print("right_count is:{}".format(right_count))
		print("total dataset is:{}".format(inference_images.get_images_number()))
		print("accuracy is:{}".format(float(right_count)/inference_images.get_images_number()))
		print("Throughput is:{}".format(batchsize*global_step*1000/float(total_time_ms)))

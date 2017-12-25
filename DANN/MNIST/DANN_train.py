import tensorflow as tf 
import numpy as np 
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from flip_gradient import flip_gradient
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 128

mnist_train = (mnist.train.images>0).reshape(55000,28,28).astype(np.uint8)*255	
temp1 = np.zeros_like(mnist_train)
mnist_train = np.stack((mnist_train,mnist_train,mnist_train),axis=3)

mnist_test = (mnist.test.images>0).reshape(10000,28,28).astype(np.uint8)*255
temp2 = np.zeros_like(mnist_test)
mnist_test = np.stack((mnist_test,mnist_test,mnist_test),axis=3)

with open('mnistm_data.pkl','rb') as f:
	data = pickle.load(f)

mnistm_train = data["train"]
mnistm_test = data["test"]

mnist_temp1 = np.tile(np.array([1,0]),(64,1))
mnist_temp2 = np.tile(np.array([0,1]),(64,1))
mnist_mix_labels = np.concatenate((mnist_temp1,mnist_temp2),axis=0)
mnist_mix = np.concatenate((mnist_train,mnistm_train),axis=0)

#p = np.random.permutation(55000)
#mnistm_train = mnistm_train[p]
#mnistm_labels= mnist.train.labels[p]

mnist_mix_test = np.concatenate((mnist_test,mnistm_test),axis=0)
mnist_temp1_test = np.tile(np.array([1,0]),(10000,1))
mnist_temp2_test = np.tile(np.array([0,1]),(10000,1))
mnist_mix_test_labels = np.concatenate((mnist_temp1_test,mnist_temp2_test),axis=0)

pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

x = tf.placeholder(tf.float32, shape=[None, 28,28,3])
y_ = tf.placeholder(tf.float32, shape=[None,10])
y2 = tf.placeholder(tf.float32, shape=[None,2])
l_param = tf.placeholder(tf.float32, shape=[])
learning_rate = tf.placeholder(tf.float32, shape=[])
training_mode = tf.placeholder(tf.bool,shape=[])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')


x_ = (tf.cast(x, tf.float32) - pixel_mean) / 255.

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

#Feature Extractor

h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 48])
b_conv2 = bias_variable([48])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*48])

#Label Classifier

all_features = lambda: h_pool2_flat
source_features = lambda: tf.slice(h_pool2_flat,[0,0],[batch_size/2,-1])
final_features = tf.cond(training_mode,source_features,all_features)

all_labels = lambda: y_
source_labels = lambda: tf.slice(y_,[0,0],[batch_size/2,-1])
final_labels = tf.cond(training_mode,source_labels,all_labels)

W_fc1 = weight_variable([7 * 7 * 48, 100])
b_fc1 = bias_variable([100])

h_fc1 = tf.nn.relu(tf.matmul(final_features, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([100, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
y_temp2 = tf.nn.softmax(y_conv)

#Domain Classifier

grl = flip_gradient(h_pool2_flat,l_param)

W_fc1_d = weight_variable([7*7*48,100])
b_fc1_d = bias_variable([100])

h_fc1_d = tf.nn.relu(tf.matmul(grl,W_fc1_d)+b_fc1_d)

W_fc2_d = weight_variable([100,2])
b_fc2_d = bias_variable([2])

y_domain = tf.matmul(h_fc1_d,W_fc2_d) + b_fc2_d

y_temp = tf.nn.softmax(y_domain)


with tf.Session() as sess:
	cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=final_labels, logits=y_conv))
	cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2, logits=y_domain))
	cross_entropy_total = cross_entropy1 + cross_entropy2
	train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy_total)
	train_step1 = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy1)
	correct_prediction1 = tf.equal(tf.argmax(y_temp2,1), tf.argmax(final_labels,1))
	correct_prediction2 = tf.equal(tf.argmax(y_temp,1), tf.argmax(y2,1))
	accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
	accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
	sess.run(tf.global_variables_initializer())
	for i in range(8000):
	  k = float(i)/ 8000.
	  l = 2. / (1. + np.exp(-10. * k)) - 1
	  #l = min(0.95,l)
	  lr = 0.01 / (1. + 10 * k)**0.75 
	  batch1_1 = mnist_train[(i%55000):((i+64)%55000),:,:,:]
	  batch2_1 = mnist.train.labels[(i%55000):((i+64)%55000),:]
	  batch1_2 = mnistm_train[(i%55000):((i+64)%55000),:,:,:]
	  batch2_2 = mnist.train.labels[(i%55000):((i+64)%55000),:]
	  batch1 = np.concatenate((batch1_1,batch1_2),axis=0)
	  batch2 = np.concatenate((batch2_1,batch2_2),axis=0)
	  if i%100 == 0:
		train_accuracy1 = accuracy1.eval(feed_dict={
			x:batch1, y_: batch2, keep_prob: 1.0, training_mode: True})
		print("step %d, training accuracy %g"%(i, train_accuracy1))
		train_accuracy2 = accuracy2.eval(feed_dict={x: batch1, y2: mnist_mix_labels, l_param: l, training_mode: True})
		print("step %d, domain classification %g"%(i, train_accuracy2))
		#batch_loss = cross_entropy_total.eval(feed_dict={x: batch1, y2: mnist_mix_labels, y_:batch2, keep_prob: 1.0, l_param: l, training_mode: True})
		#print("step %d, batch loss %g"%(i, batch_loss))
		print("learning rate ",lr)
	    train_step.run(feed_dict={x: batch1, y2: mnist_mix_labels, y_:batch2, l_param: l, learning_rate: lr, training_mode: True, keep_prob: 0.5})	
	  

	print("test accuracy source %g"%accuracy1.eval(feed_dict={
		x: mnist_test, y_: mnist.test.labels, keep_prob: 1.0, training_mode: False}))

	print("test accuracy target %g"%accuracy1.eval(feed_dict={x: mnistm_test, y_: mnist.test.labels, keep_prob: 1.0, training_mode: False}))

	print("domain accuracy %g"%accuracy2.eval(feed_dict={x: mnist_mix_test, y2:mnist_mix_test_labels, l_param: l, training_mode: False}))



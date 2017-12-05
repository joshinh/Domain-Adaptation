import tensorflow as tf 
import numpy as np 
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from flip_gradient import flip_gradient
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
mnist_train = np.reshape(mnist.train.images, (-1,28,28))
temp1 = np.zeros_like(mnist_train)
mnist_train = np.stack((mnist_train,temp1,temp1),axis=3)

mnist_test = np.reshape(mnist.test.images, (-1,28,28))
temp2 = np.zeros_like(mnist_test)
mnist_test = np.stack((mnist_test,temp2,temp2),axis=3)

with open('mnistm_data.pkl','rb') as f:
	data = pickle.load(f)

mnistm_train = data["train"]
mnistm_test = data["test"]
# mnistm_train = np.reshape(mnistm_train,(-1,784))
# mnistm_test = np.reshape(mnistm_test,(-1,784))
# mnistm_test = mnistm_test[:10000,:]

mnist_temp1 = np.tile(np.array([1,0]),(55000,1))
mnist_temp2 = np.tile(np.array([0,1]),(55000,1))
mnist_mix_labels = np.concatenate((mnist_temp1,mnist_temp2),axis=0)
mnist_mix = np.concatenate((mnist_train,mnistm_train),axis=0)

p = np.random.permutation(110000)
mnist_mix = mnist_mix[p]
mnist_mix_labels= mnist_mix_labels[p]

mnist_mix_test = np.concatenate((mnist_test,mnistm_test),axis=0)
mnist_temp1_test = np.tile(np.array([1,0]),(10000,1))
mnist_temp2_test = np.tile(np.array([0,1]),(10000,1))
mnist_mix_test_labels = np.concatenate((mnist_temp1_test,mnist_temp2_test),axis=0)

# print(mnist_mix_labels[0:10,:])

x = tf.placeholder(tf.float32, shape=[None, 28,28,3])
y_ = tf.placeholder(tf.float32, shape=[None,10])
y2 = tf.placeholder(tf.float32, shape=[None,2])

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


W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

# x_image_temp = tf.reshape(x, [-1,28,28])
# ztemp = tf.zeros(tf.shape(x_image_temp))
# x_image = tf.cond(x.shape[-1]==3, lambda:x, lambda: tf.stack([x_image_temp, ztemp, ztemp],axis=3))

# x_image = tf.stack([x_image_temp, ztemp, ztemp],axis=3)

#Feature Extractor
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#Label Classifier

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Domain Classifier

grl = flip_gradient(h_pool2_flat)

W_fc1_d = weight_variable([7*7*64,100])
b_fc1_d = bias_variable([100])

h_fc1_d = tf.nn.relu(tf.matmul(grl,W_fc1_d)+b_fc1_d)

W_fc2_d = weight_variable([100,2])
b_fc2_d = bias_variable([2])

y_domain = tf.matmul(h_fc1_d,W_fc2_d) + b_fc2_d

y_temp = tf.nn.softmax(y_domain)


with tf.Session() as sess:
	cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2, logits=y_domain))
	train_step1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy1)
	train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy2)
	correct_prediction1 = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	correct_prediction2 = tf.equal(tf.argmax(y_domain,1), tf.argmax(y2,1))
	accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
	accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
	sess.run(tf.global_variables_initializer())
	# print(mnist.test.images.shape)
	# print(mnistm_test.shape)
	for i in range(6000):
	  batch2 = mnist.train.labels[(i%55000):((i+50)%55000),:]
	  batch1 = mnist_train[(i%55000):((i+50)%55000),:,:,:]
	  if i%100 == 0:
	    train_accuracy1 = accuracy1.eval(feed_dict={
	        x:batch1, y_: batch2, keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy1))
	  train_step1.run(feed_dict={x: batch1, y_: batch2, keep_prob: 0.5})
	  if i>999:
	  	batch2_d = mnist_mix_labels[(i%110000):((i+50)%110000),:]
	  	batch1_d = mnist_mix[(i%110000):((i+50)%110000),:,:,:]
	  	if i%100 == 0:
	  		train_accuracy2 = accuracy2.eval(feed_dict={x: batch1_d, y2: batch2_d})
	  		print("step %d, domain classification %g"%(i, train_accuracy2))
	  	# print(y_temp.eval(feed_dict={x: batch1_d}))
	  	# print(y_conv.get_shape())
	  	train_step2.run(feed_dict={x: batch1_d, y2: batch2_d})	


	print("test accuracy %g"%accuracy1.eval(feed_dict={
	    x: mnist_test, y_: mnist.test.labels, keep_prob: 1.0}))

	print("domain accuracy %g"%accuracy2.eval(feed_dict={x: mnist_mix_test, y:mnist_mix_test_labels}))



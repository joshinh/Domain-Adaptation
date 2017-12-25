"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = '/home/nitish/exp/Office/finetune_alexnet_with_tensorflow/train.txt'
val_file = '/home/nitish/exp/Office/finetune_alexnet_with_tensorflow/valid.txt'
target_file = '/home/nitish/exp/Office/target_webcam.txt'
test_webcam_file = '/home/nitish/exp/Office/test.txt'

# Learning params
learning_rate = tf.placeholder(tf.float32,[])
num_epochs = 100 
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 31
train_layers = ['bottle_neck','fc8','dc1','dc2','dc3']

# How often we want to write the tf.summary data to disk
display_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
		os.makedirs(checkpoint_path)

#Create Domain labels
temp1 = np.tile(np.array([1,0]),(batch_size/2,1))
temp2 = np.tile(np.array([0,1]),(batch_size/2,1))
d_labels = np.concatenate((temp1,temp2),axis=0)		
#print(d_labels)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
		tr_data = ImageDataGenerator(train_file,
																 mode='training',
																 batch_size=batch_size/2,
																 num_classes=num_classes,
																 shuffle=True)
		val_data = ImageDataGenerator(val_file,
																	mode='inference',
																	batch_size=batch_size,
																	num_classes=num_classes,
																	shuffle=False)

		target_data = ImageDataGenerator(target_file,
																 mode='training',
																 batch_size=batch_size/2,
																 num_classes=num_classes,
																 shuffle=True)

		test_data = ImageDataGenerator(test_webcam_file,
																 mode='inference',
																 batch_size=batch_size,
																 num_classes=num_classes,
																 shuffle=False)


		# create an reinitializable iterator given the dataset structure
		iterator = Iterator.from_structure(tr_data.data.output_types,
																			 tr_data.data.output_shapes)
		next_batch = iterator.get_next()

		iterator_d = Iterator.from_structure(target_data.data.output_types, target_data.data.output_shapes)

		next_batch_d = iterator_d.get_next()

		iterator_t = Iterator.from_structure(test_data.data.output_types, test_data.data.output_shapes)

		next_batch_t = iterator_t.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

target_init_op = iterator_d.make_initializer(target_data.data)

test_init_op = iterator_t.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
l = tf.placeholder(tf.float32,[])
keep_prob = tf.placeholder(tf.float32)
tr_mode = tf.placeholder(tf.bool,[])

#Decide labels
all_labels = lambda: y
source_labels = lambda: tf.slice(y,[0,0],[batch_size/2,-1])
final_labels = tf.cond(tr_mode,source_labels,all_labels)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers, l,tr_mode)

# Link variable to model output
score = model.fc8

#Domain labels
d_pred = model.dc3


# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
print(var_list)

# Op for calculating the loss
with tf.name_scope("cross_ent"):
		loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
																																	labels=final_labels))
		loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_pred,labels=d_labels))
		loss = loss1 + loss2

# Train op
with tf.name_scope("train"):
		# Get gradients of all trainable variables
		gradients = tf.gradients(loss, var_list)
		gradients = list(zip(gradients, var_list))

		# Create optimizer and apply gradient descent to the trainable variables
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
		tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
		tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
		correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(final_labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

# Start Tensorflow session
with tf.Session() as sess:

		# Initialize all variables
		sess.run(tf.global_variables_initializer())

		# Add the model graph to TensorBoard
		writer.add_graph(sess.graph)

		# Load the pretrained weights into the non-trainable layer
		model.load_initial_weights(sess)

		print("{} Start training...".format(datetime.now()))
		print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
																											filewriter_path))

		

		# Loop over number of epochs
		for epoch in range(num_epochs):

				print("{} Epoch number: {}".format(datetime.now(), epoch+1))

				# Initialize iterator with the training dataset
				sess.run(training_init_op)
				sess.run(target_init_op)

				for step in range(train_batches_per_epoch):

						# get next batch of data

						p = ((epoch*train_batches_per_epoch)+step)/(float(num_epochs*train_batches_per_epoch))
						#l_param = 0
						l_param = 2. / (1. + np.exp(-10. * p)) - 1
						#print(l_param)
						lr = 0.01 / (1. + 10 * p)**0.75
						#lr = 0.0001
						#print(p) 

						img_batch_s, label_batch_s = sess.run(next_batch)
						img_batch_t, label_batch_t = sess.run(next_batch_d)
						img_batch = np.concatenate((img_batch_s,img_batch_t),axis=0)
						label_batch = np.concatenate((label_batch_s,label_batch_t),axis=0)

						# And run the training op
						sess.run(train_op, feed_dict={x: img_batch,
																					y: label_batch,
																					keep_prob: dropout_rate,
																					l: l_param,
																					tr_mode: True,
																					learning_rate: lr})

						# Generate summary with the current batch of data and write to file
						if step % display_step == 0:
								s = sess.run(merged_summary, feed_dict={x: img_batch,
																												y: label_batch,
																												keep_prob: 1.,
																												l: l_param,
																												tr_mode: True})

								writer.add_summary(s, epoch*train_batches_per_epoch + step)
				print("{} Start testing".format(datetime.now()))
				sess.run(test_init_op)
				test_acc = 0
				test_count = 0
		
				for _ in range(test_batches_per_epoch):

					img_batch, label_batch = sess.run(next_batch_t)
					acc = sess.run(accuracy, feed_dict={x: img_batch,
																					y: label_batch,
																					keep_prob: 1.,
																																 	     tr_mode: False})
					test_acc += acc
					test_count += 1
				test_acc /= test_count
				print("{} Webcam Test Accuracy = {:.4f}".format(datetime.now(),
																										 test_acc))
				

				# Validate the model on the entire validation set
				print("{} Start validation".format(datetime.now()))
				sess.run(validation_init_op)
				test_acc = 0.
				test_count = 0
				for _ in range(val_batches_per_epoch):

						img_batch, label_batch = sess.run(next_batch)
						acc = sess.run(accuracy, feed_dict={x: img_batch,
																								y: label_batch,
																								keep_prob: 1.,
																								tr_mode: False})
						test_acc += acc
						test_count += 1
				test_acc /= test_count
				print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
																											 test_acc))
				print("{} Saving checkpoint of model...".format(datetime.now()))

				# save checkpoint of the model
				checkpoint_name = os.path.join(checkpoint_path,
																			 'model_epoch'+str(epoch+1)+'.ckpt')
				save_path = saver.save(sess, checkpoint_name)

				print("{} Model checkpoint saved at {}".format(datetime.now(),
																											 checkpoint_name))

		print("{} Start testing".format(datetime.now()))
		sess.run(test_init_op)
		test_acc = 0
		test_count = 0
		
		for _ in range(test_batches_per_epoch):

			img_batch, label_batch = sess.run(next_batch_t)
			acc = sess.run(accuracy, feed_dict={x: img_batch,
																					y: label_batch,
																					keep_prob: 1.,
																																 	     tr_mode: False})
			test_acc += acc
			test_count += 1
		test_acc /= test_count
		print("{} Webcam Test Accuracy = {:.4f}".format(datetime.now(),
																										 test_acc))





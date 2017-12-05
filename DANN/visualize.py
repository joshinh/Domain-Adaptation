import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

with open('mnistm_data.pkl','rb') as f:
	data = pickle.load(f)

# 
# print(data["train"].shape)
# plt.imshow(data["train"][123])
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(data["test"][9999])
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(mnist.test.images[9999].reshape((28,28)))
plt.show()
print(mnist.test.labels[9999])

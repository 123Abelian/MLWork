import numpy as np
import tensorflow as tf

filename = "poyntingdata.csv"
x = tf.placeholder(tf.float32, shape=[None, 2], name='features')
weights = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(tf.zeros([1]), name="biases")
log_file = "/tmp/feature_2_batch_1000"

with tf.name_scope("Wx_b") as scope:
  dropped = tf.nn.dropout(x,0.8)
  product = tf.matmul(dropped,weights)
  y = product + b
  
y_pred = tf.placeholder(tf.float32, [None, 1])
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_pred - y))
  cost_sum = tf.summary.scalar("cost", cost)

learn_rate = 0.001
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

n_epochs = 1000
datapoint_size = 1000
batch_size = 1000
all_xs = []
all_ys = []

with open(filename) as inf:
	for line in inf:
        	poynting_coefficient, t1, t2 = line.strip().split(",")
       		poynting_coefficient = float(poynting_coefficient)
	        t1 = float(t1)
	        t2 = float(t2)
	        linelist = [t1, t2]
                all_xs.append(linelist)
                all_ys.append(poynting_coefficient)

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph_def)
init = tf.global_variables_initializer()
sess.run(init)
for i in range(n_epochs):
 if datapoint_size == batch_size:
  batch_start_idx = 0
 elif datapoint_size < batch_size:
  raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
 else:
  batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
 batch_end_idx = batch_start_idx + batch_size
 batch_xs = all_xs[batch_start_idx:batch_end_idx]
 batch_ys = all_ys[batch_start_idx:batch_end_idx]
 xs = np.array(batch_xs)
 ys = np.array(batch_ys)
 all_feed = { x: all_xs, y_pred: all_ys }
 if i % 10 == 0:
  result = sess.run(merged, feed_dict=all_feed)
  writer.add_summary(result, i)
 else:
  feed = { x: xs, y_pred: ys }
  sess.run(train_step, feed_dict=feed)
 print("After %d iterations:" % i)
 print("W: %s" % sess.run(weights))
 print("b: %f" % sess.run(b))
 print("cost: %f" % sess.run(cost, feed_dict=all_feed))


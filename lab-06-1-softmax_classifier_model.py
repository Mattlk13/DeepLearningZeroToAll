# Lab 6 Softmax Classifier
# App source: https://github.com/nalsil/TensorflowSimApp
# Play store: https://play.google.com/store/apps/details?id=com.nalsil.tensorflowsimapp
import tensorflow as tf
from utils import coldGraph
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4], name='X')
Y = tf.placeholder("float", [None, 3], name='Y')
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b, name='hypothesis')

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

argmax = tf.argmax(Y, 1, name='ud_argmax')

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('--------------')

    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))
    print('--------------')

    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

    coldGraph(sess, 'lab_06_1_softmax_classifier', "X", "hypothesis, ud_argmax", "save/Const:hypothesis, save/Const:ud_argmax" )

'''
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]] [1]
--------------
[[ 0.93119204  0.06290206  0.0059059 ]] [0]
--------------
[[  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [2]
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]
 [  9.31192040e-01   6.29020557e-02   5.90589503e-03]
 [  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [1 0 2]
'''

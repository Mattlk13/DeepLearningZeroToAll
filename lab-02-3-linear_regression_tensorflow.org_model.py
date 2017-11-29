# From https://www.tensorflow.org/get_started/get_started
# App source : https://github.com/nalsil/TensorflowSimApp
import tensorflow as tf
from utils import coldGraph

# Model parameters
W = tf.Variable([.3], tf.float32, name='W')
b = tf.Variable([-.3], tf.float32, name='b')

# Model input and output
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

hypothesis = tf.add(tf.multiply(x, W), b, name='hypothesis')
# linear_model = x * W + b

# cost/loss function
loss = tf.reduce_sum(tf.square(hypothesis - y), name='loss')  # sum of the squares

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, name='train')

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

for i in range(1000):
    # sess.run(train, {x: x_train, y: y_train})
    feed_dict = {x: x_train, y: y_train}
    sess.run([train], feed_dict=feed_dict)


# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

curr_loss, curr_hypo = sess.run([loss, hypothesis], {x: 3, y: 3})
print("loss: %s, hypothesis: %s " % (curr_loss, curr_hypo))

coldGraph(sess, 'lab_02_3_linear_regression', "x", "hypothesis", "save/Const:hypothesis" )

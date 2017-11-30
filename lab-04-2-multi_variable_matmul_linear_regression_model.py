# Lab 4 Multi-variable linear regression
# App source: https://github.com/nalsil/TensorflowSimApp
# Play store: https://play.google.com/store/apps/details?id=com.nalsil.tensorflowsimapp
import tensorflow as tf
from utils import coldGraph

tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
#hypothesis = tf.matmul(X, W) + b
hypothesis = tf.add(tf.matmul(X, W), b, name='hypothesis')

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y), name='cost')

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost, name='train')

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))


coldGraph(sess, 'lab_04_2_multi_variable_matmul_linear_regression', "X", "hypothesis", "save/Const:hypothesis" )

'''
0 Cost:  7105.46
Prediction:
 [[ 80.82241058]
 [ 92.26364136]
 [ 93.70250702]
 [ 98.09217834]
 [ 72.51759338]]
10 Cost:  5.89726
Prediction:
 [[ 155.35159302]
 [ 181.85691833]
 [ 181.97254944]
 [ 194.21760559]
 [ 140.85707092]]

...

1990 Cost:  3.18588
Prediction:
 [[ 154.36352539]
 [ 182.94833374]
 [ 181.85189819]
 [ 194.35585022]
 [ 142.03240967]]
2000 Cost:  3.1781
Prediction:
 [[ 154.35881042]
 [ 182.95147705]
 [ 181.85035706]
 [ 194.35533142]
 [ 142.036026  ]]


Your score will be  [[ 169.67449951]]
Other scores will be  [[ 105.15664673]
 [ 198.46594238]] 

'''

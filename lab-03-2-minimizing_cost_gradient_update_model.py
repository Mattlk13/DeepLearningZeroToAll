# Lab 3 Minimizing Cost
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'lab_03_2_minimizing_cost_gradient_update'
TB_SUMMARY_DIR = './saved/' + MODEL_NAME + '/'
input_graph_path = TB_SUMMARY_DIR + MODEL_NAME+'.pbtxt'
checkpoint_path = TB_SUMMARY_DIR + MODEL_NAME+'.ckpt'

tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

tf.summary.histogram("X_hist", X)
tf.summary.histogram("Y_hist", Y)

# Our hypothesis for linear model X * W
# hypothesis = X * W
hypothesis = tf.multiply(X, W, name='hypothesis')

tf.summary.histogram("hypothesis_hist", hypothesis)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y), name='cost')

tf.summary.scalar("cost_scalar", cost)

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Summary
summary = tf.summary.merge_all()
saver = tf.train.Saver()


# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# save the graph
tf.train.write_graph(sess.graph_def, TB_SUMMARY_DIR,  MODEL_NAME + '.pbtxt')

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0


for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    #print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    cost_val, W_val = sess.run([cost, W], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val, W_val)
    global_step += 1

saver.save(sess, checkpoint_path)


# =================  chkpt  ==> pb    ======
# Freeze the graph
input_saver_def_path = ""
input_binary = False
output_node_names = "cost,hypothesis"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:cost, save/Const:hypothesis"
output_frozen_graph_name = TB_SUMMARY_DIR +  'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")

# Optimize for inference
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:  # r => rb
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["X"], # an array of the input node(s)
        ["hypothesis", "cost"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(TB_SUMMARY_DIR + output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())


'''
0 1.93919 [ 1.64462376]
1 0.551591 [ 1.34379935]
2 0.156897 [ 1.18335962]
3 0.0446285 [ 1.09779179]
4 0.0126943 [ 1.05215561]
5 0.00361082 [ 1.0278163]
6 0.00102708 [ 1.01483536]
7 0.000292144 [ 1.00791216]
8 8.30968e-05 [ 1.00421977]
9 2.36361e-05 [ 1.00225055]
10 6.72385e-06 [ 1.00120032]
11 1.91239e-06 [ 1.00064015]
12 5.43968e-07 [ 1.00034142]
13 1.54591e-07 [ 1.00018203]
14 4.39416e-08 [ 1.00009704]
15 1.24913e-08 [ 1.00005174]
16 3.5322e-09 [ 1.00002754]
17 9.99824e-10 [ 1.00001466]
18 2.88878e-10 [ 1.00000787]
19 8.02487e-11 [ 1.00000417]
20 2.34053e-11 [ 1.00000226]
'''

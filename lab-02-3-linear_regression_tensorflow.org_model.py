# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'lab_02_3_linear_regression'
TB_SUMMARY_DIR = './saved/' + MODEL_NAME + '/'
input_graph_path = TB_SUMMARY_DIR + MODEL_NAME+'.pbtxt'
checkpoint_path = TB_SUMMARY_DIR + MODEL_NAME+'.ckpt'

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
tf.summary.scalar("cost", loss)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, name='train')

# Summary
summary = tf.summary.merge_all()
saver = tf.train.Saver()

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values to wrong

# save the graph
tf.train.write_graph(sess.graph_def, TB_SUMMARY_DIR,  MODEL_NAME + '.pbtxt')

# Create summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

for i in range(1000):
    # sess.run(train, {x: x_train, y: y_train})
    feed_dict = {x: x_train, y: y_train}
    s, _ = sess.run([summary, train], feed_dict=feed_dict)
    writer.add_summary(s, global_step=global_step)
    global_step += 1

saver.save(sess, checkpoint_path)

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

curr_loss, curr_hypo = sess.run([loss, hypothesis], {x: 3, y: 3})
print("loss: %s, hypothesis: %s " % (curr_loss, curr_hypo))


# =================  chkpt  ==> pb    ======
# Freeze the graph
input_saver_def_path = ""
input_binary = False
output_node_names = "hypothesis"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:hypothesis"
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
        ["x"], # an array of the input node(s)
        ["hypothesis"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(TB_SUMMARY_DIR + output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())

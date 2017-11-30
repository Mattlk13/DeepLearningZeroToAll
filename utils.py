import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


def coldGraph(sess,
              model_name,
              input_node_names,
              output_node_names,
              filename_tensor_name):

    '''
    :param sess:
    :param model_name:
    :param input_node_names:
    :param output_node_names: Enter multi-node names without spaces  ex) "hypotheis, cost" (X) ==> "hypotheis,cost"  (O)
    :param filename_tensor_name:
    :return:
    '''

    MODEL_NAME = model_name
    TB_SUMMARY_DIR = './saved/' + MODEL_NAME + '/'
    input_graph_path = TB_SUMMARY_DIR + MODEL_NAME+'.pbtxt'
    checkpoint_path = TB_SUMMARY_DIR + MODEL_NAME+'.ckpt'

    output_node_names = output_node_names.replace(' ','')

    #  Get the saver
    saver = tf.train.Saver()

    # save the graph
    tf.train.write_graph(sess.graph_def, TB_SUMMARY_DIR,  MODEL_NAME + '.pbtxt')

    # save the checkpoint
    saver.save(sess, checkpoint_path)

    # =================  chkpt  ==> pb    ======
    # Freeze the graph
    input_saver_def_path = ""
    input_binary = False
    restore_op_name = "save/restore_all"
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

    arrOutput_node_names = output_node_names.split(',')
    # print(arrOutput_node_names)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            [input_node_names], # an array of the input node(s)
            arrOutput_node_names, # an array of output nodes
            tf.float32.as_datatype_enum)

    # Save the optimized graph
    f = tf.gfile.FastGFile(TB_SUMMARY_DIR + output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

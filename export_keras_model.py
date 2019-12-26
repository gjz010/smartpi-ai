import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.framework import graph_io
K.set_learning_phase(0)

from model import model


model.load_weights("training_1/cp.ckpt")
sess=K.get_session()
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output/Softmax"])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
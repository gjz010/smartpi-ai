import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import model_from_json


K.set_learning_phase(0)
with open("model.json", "r") as file:
    config = file.read()

model = model_from_json(config)
model.load_weights("training_1/cp.ckpt")
saver=tf.compat.v1.train.Saver()
sess=K.get_session()
saver.save(sess, "./final_model")

#fw = tf.summary.FileWriter('logs', sess.graph)
#fw.close()
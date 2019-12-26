import tensorflow as tf
import numpy as np
from model import model
import os

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

f=tf.io.read_file('C:\\links\\embed\\ai\\originalset\\scissor\\thumb00359.jpg')
f=tf.image.decode_jpeg(f, channels=3)
f=tf.image.resize(f, [192, 192])
f=f-127.5
f=f/127.5
print(model.predict(tf.expand_dims(f, 0)))
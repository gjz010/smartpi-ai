import model
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

file="C:\\links\\embed\\ai\\trainset\\stone\\WIN_20191204_14_04_50_Pro.jpg"

f=tf.io.read_file(file);
f=tf.image.decode_jpeg(f);
f=((f/255)*2)-1
f=tf.image.resize(f, [192, 192])

g=cv2.imread(file)[:,:,::-1]


def predict(name):
  img=cv2.imread(file)
  img=cv2.resize(img, (192, 192))
  img=((img/255)*2)-1
  return model.model.predict(np.array([img[:,:,::-1]]))
  
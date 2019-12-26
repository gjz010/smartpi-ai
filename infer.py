import tensorflow as tf
import testset
import dataset
import realset
import os
from model import model

def change_range(image,label):
	#image=tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
	#image=tf.image.random_flip_left_right(image)
	#image=tf.image.random_flip_up_down(image)
	x=image
	return 2*image-1, label


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

#print("On training set:")
#keras_ds = dataset.ds.map(change_range).batch(64)
#print(keras_ds)
#loss,acc=model.evaluate(keras_ds, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("test mat")
v=model.predict(testset.ds.map(change_range).batch(64))
print("On test set:")		
keras_ds = testset.ds.map(change_range).batch(64)				 
loss,acc=model.evaluate(keras_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("On real set:")		
keras_ds = realset.ds.map(change_range).batch(64)				 
loss,acc=model.evaluate(keras_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
import tensorflow as tf
import dataset
import testset
import os
from model import model
from datetime import datetime
import io
import matplotlib.pyplot as plt
import numpy as np

ds=dataset.fetch(64)

@tf.function
def augment(image,label):
  image=tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
  image=tf.image.random_flip_left_right(image)
  image=tf.image.random_flip_up_down(image)
  image=tf.image.random_brightness(image, 0.05)
  image=tf.image.random_contrast(image, 0.7, 1.3)
  image=tf.keras.preprocessing.image.random_rotation(image, 30)
  image=tf.keras.preprocessing.image.random_shift(image, 0.1, 0.1)
  x=image
  return x, label
 
@tf.function
def change_range(image, label):
  #x = tf.image.random_hue(x, 0.08)
  #x = tf.image.random_saturation(x, 0.6, 1.6)
  #x = tf.image.random_brightness(x, 0.05)
  #x = tf.image.random_contrast(x, 0.7, 1.3)
  return 2*image-1, label

gen=tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=lambda x: (x/127.5)-1, width_shift_range=0.1, height_shift_range=0.1, brightness_range=(0.7, 1.3),  rotation_range=10, horizontal_flip=True, vertical_flip=True, fill_mode='constant', cval=0)

keras_ds = gen.flow_from_directory("trainset/public/Train", target_size=(192, 192), color_mode='rgb', batch_size=64,class_mode='sparse')

with open("model.json", "w") as file:
    file.write(model.to_json())
	
	
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)
# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

steps_per_epoch=tf.math.ceil(len(dataset.all_image_paths)/64).numpy()


logdir = "logs\\scalars\\current"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_dataset = tf.summary.create_file_writer(logdir + '\\dataset')


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid(tup):
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  tup=(tup[0].numpy(), tup[1].numpy())
  pred=model.predict(tup[0])
  #print(pred)
  for i in range(64):
    if(len(tup[1])<=i):
      return
    # Start next subplot.
    #plt.subplot(8, 8, i + 1, title="L=%d P=%d"%(tup[1][i], pred[i]))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #print((tup[0][i]+1)/2)
    plt.imshow((tup[0][i]+1)/2)
  
  return figure
import realset
another_ds=dataset.ds.map(change_range).batch(64)
test_ds=testset.ds.map(change_range).batch(64)
real_ds=realset.ds.map(change_range).batch(64)
def log_dataset_sample(epoch, logs):
  #if (epoch%10!=9):
  #  return
  print("start")
  #sample_t=list(another_ds.take(1))[0]
  #print(sample_t)
  #sample_i=list(test_ds.take(1))[0]
  #sample_r=list(real_ds.take(1))[0]
  print("end")
  # Prepare the plot
  #figure_t = image_grid(sample_t)
  #img_t=plot_to_image(figure_t)
  #figure_i = image_grid(sample_i)
  #img_i=plot_to_image(figure_i)
  #figure_r = image_grid(sample_r)
  #img_r=plot_to_image(figure_r)
  # Log the confusion matrix as an image summary.
  #with file_writer_dataset.as_default():
  #  tf.summary.image("Samples_Train", img_t, step=epoch)
  #  tf.summary.image("Samples_Test", img_i, step=epoch)
  #  tf.summary.image("Samples_Real", img_r, step=epoch)
  #loss,acc=model.evaluate(another_ds, verbose=2)
  #print("T Restored model, loss={:5.2f} accuracy: {:5.2f}%".format(loss, 100*acc))
  loss,acc=model.evaluate(test_ds, verbose=2)
  print("I Restored model, loss={:5.2f} accuracy: {:5.2f}%".format(loss, 100*acc))
  loss,acc=model.evaluate(real_ds, verbose=2)
  print("R Restored model, loss={:5.2f} accuracy: {:5.2f}%".format(loss, 100*acc))

# Define the per-epoch callback.
dataset_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_dataset_sample)


model.fit_generator(keras_ds, epochs=20000, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback, tensorboard_callback, dataset_callback])


print("On training set:")
keras_ds = dataset.ds.map(change_range).batch(64)
loss,acc=model.evaluate(keras_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
model.save_weights("weights.h5");
#ev=model.evaluate(ds, steps=10)
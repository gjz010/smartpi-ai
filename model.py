import tensorflow as tf
import os
mobile_net=tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=True


#image_batch, label_batch = next(iter(keras_ds))
#feature_map_batch = mobile_net(image_batch)
model = tf.keras.Sequential([
	mobile_net,
	tf.keras.layers.GlobalAveragePooling2D(),
	tf.keras.layers.Dense(4, activation = 'softmax', name='output')])
	

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
print(model.outputs)
model.summary()


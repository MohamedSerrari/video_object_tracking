import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.utils import get_confusion_matrix


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

class_names = ["Bowl", "CanOfCocaCola", "MilkBottle", "Rice", "Sugar"]

train_directory = 'DB/train'
test_directory = 'DB/test'

batch_size = 16

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.9,1.0],
        fill_mode="nearest")



test_datagen = ImageDataGenerator(rescale=1./255)

# Data loading 
train_batches = train_datagen.flow_from_directory(
        train_directory,
        target_size=(224,224),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True)

test_batches = test_datagen.flow_from_directory(
        test_directory,
        target_size=(224,224),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=False)

found_classes = list(train_batches.class_indices.keys())
print('Classes Found:', found_classes)

assert all(a == b for a, b in zip(found_classes, class_names)), 'Found classes are different than static classes names\
                                                                please modify class_names in python file'
        
# Model architecture 
mobilenet = MobileNetV2(weights = 'imagenet', include_top = True, input_shape=(224,224,3))
mobilenet.layers.pop()

# for layer in mobilenet.layers : 
#     layer.trainable = False 

model = Sequential()
model.add(mobilenet)
model.add(Dense(5, activation='softmax', name='predictions'))
model.summary()

# Model training
num_train = 4736
num_val = 3568
num_epoch = 25

## Call backs
stamped_dir = 'logs/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

checkpoint_dir = stamped_dir + '/model_checkpoints'
model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
        )

logdir = stamped_dir + '/scalars/'

tensorboard_callback = TensorBoard(log_dir=logdir)

# # Learning rate callback
# def scheduler(epoch, lr):
#     if epoch < 5:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1 * epoch)

# lr_callback = LearningRateScheduler(scheduler)

# Confusion matrix callback
logdir = stamped_dir + '/image'
file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        y_pred_raw = model.predict(test_batches)
        y_pred = np.argmax(y_pred_raw, axis=1)
        y_true = test_batches.classes

        # Get the confusion
        cm_image = get_confusion_matrix(y_true, y_pred, class_names=class_names, normalize='true')

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
                tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
cm_callback = LambdaCallback(on_epoch_end=log_confusion_matrix)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, decay=1e-6), metrics=['accuracy', 'mae', 'mse'])
model.fit(
        train_batches,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=test_batches,
        validation_steps=num_val // batch_size,
        callbacks=[model_checkpoint_callback, tensorboard_callback, cm_callback]
        )
        
model.save(stamped_dir + '/model.h5')




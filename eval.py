import numpy as np
import tensorflow as tf
from utils.utils import create_model
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(device_lib.list_local_devices())

class_names = ["Bowl", "CanOfCocaCola", "MilkBottle", "Rice", "Sugar"]

batch_size = 64
test_directory = 'DB/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_batches = test_datagen.flow_from_directory(
        test_directory,
        target_size=(224,224),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=False)


checkpoint_dir = 'logs/11-12-2020-19-47-22/model_checkpoints'

model = create_model()
model.load_weights(checkpoint_dir)

y_pred_raw = model.predict(test_batches)
y_pred = np.argmax(y_pred_raw, axis=1)
y_true = test_batches.classes

print(classification_report(y_true, y_pred, target_names=class_names))




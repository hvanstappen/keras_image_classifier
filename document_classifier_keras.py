# Classify images, based on training data
#
# Usage:
# 1. create folder with:
#    - folder with training data (one folder for each type)
#    - folder with images to be classified
#    - this script
# 3. set required parameters:
#    - data_dir = (relative) folder with traing/validation images ('document_images')
#    - epoch = number of passes of the entire training dataset in the machine learning algorithm ('10')
#    - path = (relative) folder with images that need to be predicted ('test')
# 3. in terminal: '$ python document_classifier_keras.py -d data_dir -p path [-e 10] '
# 4. results are written to csv file 'predicted_image_types.csv'

# see https://www.tensorflow.org/tutorials/images/classification

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import argparse

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", default="document_images",
	help="path to traing images")
ap.add_argument("-p", "--path", default="path",
	help="path to input images")
ap.add_argument("-e", "--epoch", default="10", type=int,
	help="number of epochs")
args = vars(ap.parse_args())

path = args["path"]
data_dir = args["data_dir"]
epoch = args["epoch"]

data_dir = pathlib.Path(data_dir)
subfolders = os.listdir(data_dir)
num_classes = len(subfolders)

# Check if files are valif jpg
print("Reading and checking files from subfolders: ", subfolders, " in ", data_dir)
print("no. of subfolders: ",num_classes)

# Filter out corrupted images
# Change folder names accordingly
num_skipped = 0
for folder_name in subfolders:
    folder_path = os.path.join(data_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
            print("- Deleted file ", fpath)

print("Deleted %d images" % num_skipped)

# list no. of files
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total no of images: ", image_count)

# Create a dataset
# Define some parameters for the loader
batch_size = 32
img_height = 180
img_width = 180

# Create a validation split: 80% of the images for training, and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print("class_names: ", class_names)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize the data
# Create the model

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Train the model
epochs=15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# No optimization necessary; check tutorial if it is (eg. solve overfitting)

# Predict on new data
path = "test"
files = os.listdir(path)

# Create csv with predictions
csv = open('predicted_image_types.csv','w')

for f in files:
	f = path+'/'+f

	img = keras.preprocessing.image.load_img(
	    f, target_size=(img_height, img_width)
	)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	print(
	    "Image {} most likely belongs to {} with a {:.2f} percent confidence."
	    .format(f, class_names[np.argmax(score)], 100 * np.max(score))
	)

	# write result per image
	csv.write(str(f))
	csv.write(";")
	csv.write(class_names[np.argmax(score)])
	csv.write(";")
	csv.write(str(100 * np.max(score)))
	csv.write("\n")

print("Done. Processed all ", image_count, " images")

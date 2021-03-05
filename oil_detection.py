# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
import os.path
import re
import hashlib
from tensorflow.python.util import compat

# Image manipulation https://auth0.com/blog/image-processing-in-python-with-pillow/
from PIL import Image

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
testing_percentage = 10
validation_percentage = 10

width = 720
height = 1280


# Gather the images in the olive oil photos folder and make them into something tensor flow can understand
def create_image_lists():
    image_dir = "oliveoilphotos"
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    # The root directory comes first, so skip it.
    is_root_dir = True  # TODO

    training_images = []
    training_labels = []

    testing_images = []
    testing_labels = []

    for sub_dir in sub_dirs:
        extensions = ['png', 'jpg', 'jpeg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            print("root dir, will skip")
            continue

        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))

        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # ex: IMG_2223.png

            # Resize all the images so they are manageable
            image = Image.open("./" + file_name).convert('LA')
            image.thumbnail((height, width))
            image.save("./" + file_name)
            image_array = np.array(image)[:,:,0]
            #image_array = image_array.transpose(2, 0, 1).reshape(3, -1)
            print(image_array.shape)

            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
                testing_labels.append(dir_name)
            else:
                training_images.append(image_array)
                training_labels.append(dir_name)

        result[dir_name] = {
            'dir': dir_name,
            'training': np.array(training_images),
            'testing': np.array(testing_images),
            'validation': np.array(validation_images),
        }
    return (np.array(training_images), training_labels), (testing_images, testing_labels)

#fashion_mnist = tf.keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = create_image_lists()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(type(train_images))
print(train_images.shape)
print(train_labels)
print(tf.__version__)

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
# It's important that the training set and the testing set be preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(width, height)),  # Make 1D array of pixels
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  # Array or arrays of which ones it should be
np.argmax(predictions[0])
print(test_labels[0])

# test single image on the model
img = test_images[1]
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)

print(predictions_single)
np.argmax(predictions_single[0])

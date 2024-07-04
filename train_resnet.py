from IPython.display import clear_output
import logging
import os
import urllib.request
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
import pandas as pd
import seaborn as sns
DATA_PATH = "TB_Chest_Radiography_Database/"

BATCH_SIZE       = 32
IMG_HEIGHT_WIDTH = 256
IMG_INPUT_SHAPE  = (IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH, 3)
MAX_EPOCHS       = 30

DS_TRAIN = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH),
    batch_size=BATCH_SIZE
)
DS_VALID = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH),
    batch_size=BATCH_SIZE
)

AUTOTUNE = tf.data.AUTOTUNE
DS_TRAIN = DS_TRAIN.cache().prefetch(buffer_size=AUTOTUNE)
DS_VALID = DS_VALID.cache().prefetch(buffer_size=AUTOTUNE)
print("Start")
def get_model():
    model = None
 
# Set the local path where you want to save the downloaded file
    local_path = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    url = "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    # Download the file and skip hash validation
    tf.keras.utils.get_file(
        fname=local_path,
        origin=url,
        cache_dir=None,
        cache_subdir="",
        file_hash=None,
        hash_algorithm=None,
        extract=False
    )

    model = tf.keras.applications.ResNet50(include_top=False, weights=local_path, input_shape=IMG_INPUT_SHAPE)
    return model

# Define the model architecture
BASIC_MODEL = tf.keras.models.Sequential()
BASIC_MODEL.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
BASIC_MODEL.add(tf.keras.layers.MaxPooling2D((2, 2)))
BASIC_MODEL.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
BASIC_MODEL.add(tf.keras.layers.MaxPooling2D((2, 2)))
BASIC_MODEL.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
BASIC_MODEL.add(tf.keras.layers.Flatten())
BASIC_MODEL.add(tf.keras.layers.Dense(256, activation='relu'))
BASIC_MODEL.add(tf.keras.layers.Dense(64, activation='relu'))
BASIC_MODEL.add(tf.keras.layers.Dense(2, activation='softmax'))

# Call the function to get the ResNet50 model
RESNET50_MODEL = get_model()

# Get the output tensor of the ResNet50 model
resnet_output = RESNET50_MODEL.output

# Add a pooling layer to match the output shape of ResNet50
pooled_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)

# Concatenate the outputs from BASIC_MODEL and RESNET50_MODEL
merged_output = tf.keras.layers.Concatenate()([BASIC_MODEL.output, pooled_output])

# Build the final model
final_model = tf.keras.models.Model(inputs=[BASIC_MODEL.input, RESNET50_MODEL.input], outputs=merged_output)

# Compile and fit the model
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Save the model
final_model.save('RESNET_model.h5')



from IPython.display import clear_output: This imports the clear_output function from the IPython.display module. It is used to clear the output of Jupyter Notebook cells.

import logging: This imports the logging module, which is used for logging messages.

import os: This imports the os module, which provides functions for interacting with the operating system.

import urllib.request: This imports the urllib.request module, which is used for handling HTTP requests and downloading files.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3': This sets an environment variable to suppress TensorFlow's log messages with a severity level of INFO or lower. A level of '3' corresponds to FATAL.

logging.getLogger('tensorflow').setLevel(logging.FATAL): This sets the logging level of the TensorFlow module to FATAL, effectively suppressing TensorFlow's log messages.

import tensorflow as tf: This imports the TensorFlow library, a popular open-source machine learning framework.

import pandas as pd: This imports the pandas library, which provides data manipulation and analysis tools.

import seaborn as sns: This imports the seaborn library, which is used for data visualization.

DATA_PATH = "TB_Chest_Radiography_Database/": This sets the path to the dataset directory. You need to replace this with the actual path to your dataset.

BATCH_SIZE = 32: This sets the batch size for training and evaluation of the model.

IMG_HEIGHT_WIDTH = 256: This sets the height and width of the input images for the model.

IMG_INPUT_SHAPE = (IMG_HEIGHT_WIDTH, IMG_HEIGHT_WIDTH, 3): This defines the input shape of the model as a tuple with height, width, and channels. In this case, the images are assumed to have three channels (RGB).

MAX_EPOCHS = 30: This sets the maximum number of training epochs for the model.

DS_TRAIN = tf.keras.utils.image_dataset_from_directory(: This function call creates a TensorFlow dataset from the images in the specified directory for training.

DS_VALID = tf.keras.utils.image_dataset_from_directory(: This function call creates a TensorFlow dataset from the images in the specified directory for validation.

AUTOTUNE = tf.data.AUTOTUNE: This sets the buffer size for prefetching data to optimize performance.

DS_TRAIN = DS_TRAIN.cache().prefetch(buffer_size=AUTOTUNE): This caches and prefetches the training dataset to optimize performance.

DS_VALID = DS_VALID.cache().prefetch(buffer_size=AUTOTUNE): This caches and prefetches the validation dataset to optimize performance.

print("Start"): This prints the message "Start".

def get_model():: This defines a function named get_model() that will be used to retrieve the ResNet50 model.

local_path = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5": This sets the local path where the downloaded ResNet50 model weights will be saved.

url = "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5": This is the URL from where the ResNet50 model weights will be downloaded.

tf.keras.utils.get_file(: This function call downloads the ResNet50 model weights from the specified URL and saves them to

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    fp = "./model/RESNET_model.h5"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# X-Ray Classification [Tuberculosis/Normal] using ResNet Architecture

""")


temp = st.file_uploader("Upload X-Ray Image")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")

else:

    img = image.load_img(temp_file.name, target_size=(
        500, 500), color_mode='grayscale')

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/255
    pp_img = np.expand_dims(pp_img, axis=0)

    # predict
    preds = cnn.predict(pp_img)
    if preds >= 0.5:
       # out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(
          #  preds[0][0]))
        out = ('We are pretty confirmed that this is a Tuberculosis case'.format(
            preds[0][0]))

    else:
       # out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(
          #  1-preds[0][0]))
        out = ('We are pretty confirmed that this is a Normal case'.format(
            1-preds[0][0]))

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)



import streamlit as st: This imports the Streamlit library for building interactive web applications.

from PIL import Image: This imports the Image module from the Python Imaging Library (PIL) for image processing.

from tensorflow.keras.models import load_model: This imports the load_model function from tensorflow.keras.models module for loading a pre-trained model.

import tensorflow as tf: This imports the TensorFlow library.

from tempfile import NamedTemporaryFile: This imports the NamedTemporaryFile class from the tempfile module to create a temporary file for image upload.

from tensorflow.keras.preprocessing import image: This imports the image module from tensorflow.keras.preprocessing for image preprocessing.

st.set_option('deprecation.showfileUploaderEncoding', False): This disables a Streamlit warning related to file uploader encoding.

@st.cache(allow_output_mutation=True): This is a decorator that enables caching for the loading_model function, allowing it to be executed only once.

def loading_model():: This is a function that loads the pre-trained ResNet model.

fp = "./model/RESNET_model.h5": This sets the file path to the pre-trained model. You need to replace this with the actual file path.

model_loader = load_model(fp): This loads the pre-trained model using the load_model function and assigns it to the model_loader variable.

return model_loader: This returns the loaded model.

cnn = loading_model(): This calls the loading_model function to load the model and assigns it to the cnn variable.

st.write("""\n# X-Ray Classification [Tuberculosis/Normal] using ResNet Architecture\n"""): This displays a title using st.write().

temp = st.file_uploader("Upload X-Ray Image"): This creates a file uploader component in the Streamlit app, allowing users to upload an X-Ray image.

buffer = temp: This assigns the uploaded file to the buffer variable.

temp_file = NamedTemporaryFile(delete=False): This creates a named temporary file to store the uploaded image.

if buffer:: This checks if an image file has been uploaded.

temp_file.write(buffer.getvalue()): This writes the contents of the uploaded file to the temporary file.

st.write(image.load_img(temp_file.name)): This displays the uploaded image using st.write() and the load_img() function from PIL.

if buffer is None:: This checks if no image file has been uploaded.

st.text("Oops! that doesn't look like an image. Try again."): This displays an error message using st.text().

else:: This is the start of the else block, which is executed when an image file has been uploaded.

img = image.load_img(temp_file.name, target_size=(500, 500), color_mode='grayscale'): This loads the uploaded image, resizes it to the target size of (500, 500), and converts it to grayscale.

pp_img = image.img_to_array(img): This converts the PIL image to a NumPy array.

pp_img = pp_img/255: This normalizes the pixel values of the image.

`pp_img = np.expand_dims
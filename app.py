import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


import tensorflow as tf
import pickle
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences


# # load vgg16 model
# model = VGG16()
# # restructure the model
# model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# load features from pickle
with open(os.path.join('features.pkl'), 'rb') as f:
    features = pickle.load(f)

# load tokenizer file
tokenizer = pickle.load(open("tokenizer.p","rb"))

# get maximum length of the caption available
max_length = 35


# load the model
from tensorflow import keras
model = keras.models.load_model('best_model_45.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Step 4: Define the image captioning function

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None




# generate caption for an image
# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text[8:-6]

def generate_image_caption(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)

    # Generate features from the image using a pre-trained VGG16 model
    vgg16 = VGG16()
    vgg16 = tf.keras.Model(inputs=vgg16.inputs, outputs=vgg16.layers[-2].output)
    features = vgg16.predict(image)

    # Generate caption using the pre-trained image captioning model
    max_length = 35
    caption = predict_caption(model, features, tokenizer, max_length)

    return caption

# Step 6: Create the Streamlit web application
st.title('Image Caption Generator')

uploaded_file = st.file_uploader('Upload an image')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Generate Caption'):
        caption = generate_image_caption(image)
        st.write('**Generated Caption:**', caption)

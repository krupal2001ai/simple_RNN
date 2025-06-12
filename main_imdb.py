
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore

import requests
import pickle



try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("TensorFlow not installed! Check requirements.txt")



# 2. Download the word index (same as imdb.get_word_index())
word_index_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"
response = requests.get(word_index_url)
word_index = response.json()

# 3. Create reverse mapping (integer → word)
reverse_word_index = {value: key for key, value in word_index.items()}

# load the pretrain model
model = load_model('imdb_with_simple_RNN.keras')
model.summary()


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    padded_review = tf.keras.utils.pad_sequences([encoded_review], maxlen=500)
    return padded_review



import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
# from tensorflow.keras.utils import pad_sequences
import requests
import pickle



#  Download the word index (same as imdb.get_word_index())
word_index_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json"
response = requests.get(word_index_url)
word_index = response.json()

#  Create reverse mapping (integer → word)
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('D:\dream\ml\ml_p\imdb_with_simple_RNN.keras')
model.summary()

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = tf.keras.utils.pad_sequences([encoded_review], maxlen=500)
    return padded_review

### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]


# Step 4: User Input and Prediction
# Example review for prediction
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."

sentiment,score=predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {score}')


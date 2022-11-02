from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import tensorflow as tf
import random
import json
import numpy as np
import pickle
import requests
import json
import os
import re
import random
import time
from keras.models import load_model


'''
# get value from enviroment variable
tenorflow_url = os.environ.get(
    'TENSORFLOW_URL', 'http://localhost:1000/v1/models/predictive_text:predict')

predict_threshold = os.environ.get(
    'pred_threshold', "0.2")

predict_threshold = float(predict_threshold)
# Get responce from tensorflow model server
'''

new_model = tf.keras.models.load_model('my_model.h5')
vocabulary=['\n', '\r', ' ', '!', '&', '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '£', '½', '‘', '’', '“', '”']

index_2_vocab_arr=np.array(vocabulary)

vocab_2_index= {'\n': 0, '\r': 1, ' ': 2, '!': 3, '&': 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '?': 22, 'A': 23, 'B': 24, 'C': 25, 'D': 26, 'E': 27, 'F': 28, 'G': 29, 'H': 30, 'I': 31, 'J': 32, 'K': 33, 'L': 34, 'M': 35, 'N': 36, 'O': 37, 'P': 38, 'Q': 39, 'R': 40, 'S': 41, 'T': 42, 'U': 43, 'V': 44, 'W': 45, 'X': 46, 'Y': 47, 'Z': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, '£': 75, '½': 76, '‘': 77, '’': 78, '“': 79, '”': 80}



def generate_text(model, start_string, num_generate, temperature):
  start = time.time()
  #Converting the input string to vector
  input_eval = [vocab_2_index[s] for s in start_string]
  #Converting into required tensor dimension
  input_eval = tf.expand_dims(input_eval, 0) 
  #Empty string to store the predictions
  text_generated = [] 
  # Clears the hidden states in the RNN
  model.reset_states() 

  for i in range(num_generate): 
    # prediction for single character
    predictions = model(input_eval) 
    predictions = tf.squeeze(predictions, 0) 
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    #Here we are taking the predicted char as the next input to the model
    input_eval = tf.expand_dims([predicted_id], 0) 
    # Also devectorize the number and add to the generated text
    text_generated.append(index_2_vocab_arr[predicted_id]) 
  end = time.time()

  global time_taken
  time_taken = end-start

  print(start_string + ''.join(text_generated))
  print('\n\nRun time took by this model:', time_taken)

# function to clean the word of any punctuation or special characters and lowwer it





def chatbot_response(msg):
    pred = generate_text(
                    new_model, 
                    num_generate = 1000, 
                    temperature = 0.5, 
                    start_string = msg
                    )
    return pred


app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    run_with_ngrok(app)
    app.run()

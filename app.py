from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import random
import json
import numpy as np
import pickle
import requests
import json
import os
import re


# get value from enviroment variable
tenorflow_url = os.environ.get(
    'TENSORFLOW_URL', 'http://localhost:1000/v1/models/predictive_text:predict')

predict_threshold = os.environ.get(
    'pred_threshold', "0.2")

predict_threshold = float(predict_threshold)
# Get responce from tensorflow model server





def get_responce_from_model_server(msg):
    input_ids = tokenizer.encode(msg, return_tensors='tf')
    data = json.dumps(
        {"signature_name": "serving_default", "instances": input_ids.numpy().tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        tenorflow_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)
    return predictions



def genre_predictor(input_ids):
  gpt_generate = gpt_2_p_model.generate(
                                input_ids,
                                do_sample = True, 
                                max_length = 2*MAX_LEN,#to test how long we can generate and it be coherent
                                top_k = 50, 
                                top_p = 0.85, 
                                num_return_sequences = 1
  )


  for i, sample_output in enumerate(gpt_generate):
      print(tokenizer.decode(sample_output, skip_special_tokens = True))
      print('')

# function to clean the word of any punctuation or special characters and lowwer it





def chatbot_response(msg):
    pred = get_responce_from_model_server(msg)
    pred = genre_predictor(pred)
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

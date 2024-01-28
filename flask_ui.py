from flask import Flask, render_template, request
import os
from PIL import Image
import requests
import numpy as np
from dotenv import load_dotenv
from ml_scripts.MongoDBDataset import get_vocab
from io import BytesIO
import logging
import re
import base64
from urllib.parse import urlparse

logging.getLogger().setLevel(logging.DEBUG)


app = Flask(__name__)

# Define the address of your TensorFlow Serving container
TF_SERVING_URL = "http://localhost:8510/v1/models/flower:predict"
load_dotenv(override=True)
category_flower_mapping = {value: key for key, value in get_vocab("flower.db").items()}


# Function to convert image to numpy array
def process_image(image_url):
    logging.info("Loading image and creating array.")
    parsed_url = urlparse(image_url)

    # Check if the URL is a data URL
    if parsed_url.scheme == "data":
        # Extract base64-encoded image data from the URL
        image_data = re.sub("^data:image/.+;base64,", "", image_url)
        image_binary = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_binary))
    else:
        # Open the image from a regular URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    # Resize or preprocess the image as needed
    image_array = np.array(image)
    return image_array


# Function to make a prediction using TensorFlow Serving
def make_prediction(image_array):
    logging.info("making prediction")
    payload = {"instances": [image_array.tolist()]}
    response = requests.post(TF_SERVING_URL, json=payload)
    predictions = response.json()["predictions"]
    predicted_class = category_flower_mapping[np.argmax(predictions[0])]
    return predicted_class


# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")


# Route for handling image upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    image_url = request.form.get("image_url")

    if not image_url:
        return render_template("index.html", error="No image URL provided")

    try:
        # Process the image and make a prediction
        image_array = process_image(image_url)
        predicted_class = make_prediction(image_array)

        # Render the result page with the image and predicted class
        return render_template("result.html", image_url=image_url, predicted_class=predicted_class)

    except Exception as e:
        return render_template("index.html", error=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    app.run("0.0.0.0")

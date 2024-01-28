import base64
import logging
import re
from io import BytesIO
from urllib.parse import urlparse

import sqlite3
import numpy as np
import requests
import streamlit as st

from dotenv import load_dotenv
from PIL import Image

from ml_scripts.MongoDBDataset import get_vocab
import polars as pl

logging.getLogger().setLevel(logging.DEBUG)

# Define the address of your TensorFlow Serving container
TF_SERVING_URL = "http://localhost:8510/v1/models/flower:predict"
load_dotenv(override=True)


def submit():
    """Submits the url."""
    st.session_state.image_url = st.session_state.url_widget
    st.session_state.url_widget = ""


@st.cache_data
def process_image(image_url: str) -> np.ndarray:
    """Converts an image to a numpy array.

    Args:
        image_url: The URL of the image to convert.

    Returns:
        np.ndarray: A numpy array of the image.
    """
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
@st.cache_data
def make_prediction(image_array: np.ndarray, category_flower_mapping: dict) -> str:
    """Makes a prediction using TensorFlow Serving.

    Args:
        image_array: A numpy array of the image to classify.
        category_flower_mapping: A dictionary mapping category to flower type.

    Returns:
        str: The predicted class of the image.
    """
    logging.info("making prediction")
    payload = {"instances": [image_array.tolist()]}
    response = requests.post(TF_SERVING_URL, json=payload)
    predictions = response.json()["predictions"]
    predicted_class = category_flower_mapping[np.argmax(predictions[0])]
    return predicted_class


def check_if_correct(conn: sqlite3.Connection, predicted_class: str):
    """Checks if the prediction is correct, and writes to the database.

    Args:
        conn: A connection to the SQLite database.
        image_url: The URL of the image that was classified.
        predicted_class: The predicted class of the image.
    """
    # display result
    with st.form("Results"):
        st.image(st.session_state.image_url, caption=f"Predicted class: {predicted_class}")
        correct_prediction = st.checkbox("Is this correct? The please check this box.", value=False)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Thank you for your feedback! Here's a flower for you 🌸")

            # write feedback to database if not previously contained for this url
            insert_statement = """
            INSERT OR REPLACE INTO feedback (image_url, predicted_class, correct_prediction)
            VALUES (?, ?, ?)
            """
            c = conn.cursor()
            c.execute(
                insert_statement,
                (st.session_state.image_url, predicted_class, correct_prediction),
            )
            conn.commit()


@st.cache_resource
def setup():
    """Creates the database table if it doesn't already exist."""
    category_flower_mapping = get_vocab("flower.db", reverse=True)

    conn = sqlite3.connect("flower.db", check_same_thread=False)
    # load schema from feedback_schema.sql file
    with open("feedback_schema.sql") as feedback_schema_file:
        create_table_statement = feedback_schema_file.read()

    c = conn.cursor()
    c.execute(create_table_statement)
    conn.commit()

    if "image_url" not in st.session_state:
        st.session_state.image_url = ""

    return conn, category_flower_mapping


def main():
    """Builds the UI for the app."""
    st.title("Flower Classifier")
    st.write("This app uses a TensorFlow model to classify images of flowers.")
    sql_conn, category_flower_mapping = setup()

    st.text_input("Enter the URL of an image:", key="url_widget", value="", on_change=submit)
    if st.session_state.image_url:
        image_array = process_image(st.session_state.image_url)
        predicted_class = make_prediction(image_array, category_flower_mapping)
        check_if_correct(sql_conn, predicted_class)


if __name__ == "__main__":
    main()

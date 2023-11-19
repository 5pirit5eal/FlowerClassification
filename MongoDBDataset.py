import os
from re import I
from dotenv import load_dotenv
from pymongo import MongoClient
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import sys
import random

load_dotenv()


def features_and_labels(collection, mode="train"):
    id_list = [
        mongodb_document["id"]
        for mongodb_document in collection.find({}, projection={"_id": False, "id": True}).limit(10)
    ]

    def features_and_labels_generator():
        if mode == "train":
            random.shuffle(id_list)
        for document_id in id_list:
            mongodb_document = collection.find_one({"id": document_id})
            label = mongodb_document["category"]
            features = Image.open(io.BytesIO(mongodb_document["data"]))
            yield features, label

    return features_and_labels_generator


@tf.function
def conditional_grayscale_to_rgb(image, label):
    if image.shape[-1] == 1 and tf.rank(image) == 3:
        # tf.print(image.shape, output_stream=sys.stdout)
        return tf.image.grayscale_to_rgb(image), label
    elif tf.rank(image) < 3:
        return tf.image.grayscale_to_rgb(tf.expand_dims(image, axis=-1)), label
    elif tf.rank(image) > 3:
        return tf.squeeze(image), label
    else:
        return image, label


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.int64, name="data"),
        tf.TensorSpec(shape=(), dtype=tf.int8, name="category"),
    )
)
def resize_and_pad(image, label):
    image = tf.image.resize(image, (256, 256), preserve_aspect_ratio=True)
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    return image, label


def create_dataset(collection, batch_size=3, mode="eval"):
    dataset = tf.data.Dataset.from_generator(
        features_and_labels(collection),
        output_signature=(
            tf.TensorSpec(shape=None, dtype=tf.int64, name="data"),
            tf.TensorSpec(shape=(), dtype=tf.int8, name="category"),
        ),
    )
    n_classes = len(collection.distinct("category"))
    # dataset = dataset.map(load_image_from_bytes, name="load_image")
    dataset = dataset.map(conditional_grayscale_to_rgb, name="gray2rgb")
    dataset = dataset.map(resize_and_pad, name="resize")
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, n_classes)), name="categorize_label")

    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


if __name__ == "__main__":
    CLIENT = MongoClient(os.environ["URI"])
    DB = CLIENT.flowers
    dataset = create_dataset(DB.train, mode="train")
    for data, label in dataset:
        print(data.shape)
        print(label.shape)

    CLIENT.close()

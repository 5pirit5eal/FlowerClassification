import io
import os
import random
import sqlite3 as sql
import time

import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient

load_dotenv()


def create_vocab():
    CLIENT = MongoClient(os.environ["URI"])
    DB = CLIENT.flowers
    collection = DB.test
    flower_types = collection.distinct("flower_type")

    # Create a mapping of flower_type to category
    flower_type_mapping = {}
    for flower_type in flower_types:
        flower = collection.find_one({"flower_type": flower_type})
        flower_type_mapping[flower_type] = flower.get("category", None)

    CLIENT.close()

    return flower_type_mapping


def save_vocab(sql_uri: str):
    vocab = create_vocab()
    if not os.path.exists(sql_uri):
        # create database file
        open(sql_uri, "w").close()

    with sql.connect(sql_uri) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS flower_vocab (flower_type text, category int)")
        for flower_type, category in vocab.items():
            c.execute("INSERT INTO flower_vocab VALUES (?, ?)", (flower_type, category))
        conn.commit()


def get_vocab(sql_uri: str, reverse: bool = True):
    with sql.connect(sql_uri) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM flower_vocab ORDER BY category ASC")
        vocab = {flower_type: category for flower_type, category in c.fetchall()}
        if reverse:
            vocab = {category: flower_type for flower_type, category in vocab.items()}
    return vocab


def features_and_labels(collection, mode="train"):
    id_list = [
        mongodb_document["_id"]
        for mongodb_document in collection.find({}, projection={"_id": True})
    ]

    def features_and_labels_generator():
        if mode == "train":
            random.shuffle(id_list)
        for document_id in id_list:
            mongodb_document = collection.find_one({"_id": document_id})
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


def create_dataset(collection, batch_size=3, mode="eval") -> tf.data.Dataset:
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
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    save_vocab("flower.db")
    # CLIENT = MongoClient(os.environ["URI"])
    # DB = CLIENT.flowers
    # time.sleep(10)

    # dataset = create_dataset(DB.train, mode="train")
    # for data, label in dataset:
    #     print(data.shape)
    #     print(label.shape)

    # CLIENT.close()

# -*- coding: utf-8 -*-
"""NER-Transformer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Frg-3Nyr6Sy2JnpqElQj7iY3JRWwtlqc

# Named Entity Recognition with Transformer
Named Entity Recognition dataset using Transformer model to evaluate student writing [Evaluating Student Writing]
## Import Packages
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict
from tensorflow.keras import layers
from collections import defaultdict

"""## Utilities

## Model Parameters
"""

vocab_size = 10000 # Vocabulary size
sequence_length = 1024 # Sequence Length
batch_size = 128 # Batch size
unk_token = "<UNK>" # Unknownd token
padding_token = "<PAD>"
none_class = "O"
vectorizer_path = "vectorizer.json"
# Use output dataset for inference
output_dataset_path = "/shahbaz/NER-output/"
model_path = "model.h5"
embed_size = 128
hidden_size = 64
modes = ["training", "inference"] # There is training and inference mode
mode = modes[0]
epochs = 30

"""## Import Datasets"""

train = pd.read_csv("shahbaz/training/train.csv")
train.head()

submission = pd.read_csv("shahbaz/training/sample_submission.csv")
submission.head()

"""## EDA & Preprocessing

### Add File Path to train and submission Files
"""

train["file_path"] = train["id"].apply(lambda item: "shahbaz/training/train/" + item + ".txt")
train.head()

submission["file_path"] = submission["id"].apply(lambda item: "shahbaz/testing/test/" + item + ".txt")
submission.head()

"""## Distribution of Labels"""

train["discourse_type"].value_counts().plot(kind="bar")

discourse_types = list(train["discourse_type"].value_counts().index)

all_tags = [padding_token]
for discourse_type in discourse_types:
    all_tags.append("B-" + discourse_type)
for discourse_type in discourse_types:
    all_tags.append("I-" + discourse_type)
all_tags.append(none_class)
pad_index = all_tags.index(padding_token)
none_index = all_tags.index(none_class)
tag_index = dict([(tag, index) for (index, tag) in enumerate(all_tags)])
index_tag = dict([(tag_index[tag], tag.replace("B-", "").replace("I-", "")) for tag in tag_index])
print(all_tags)
print(tag_index)
print(index_tag)

"""### Number of Unique files"""

len(train["id"].unique())

"""### Tokenization"""

def tokenize(content):
    tokens = content.lower().split()
    return tokens

def calc_word_indices(full_text, discourse_start, discourse_end):
    start_index = len(full_text[:discourse_start].split())
    token_len = len(full_text[discourse_start:discourse_end].split())
    output = list(range(start_index, start_index + token_len))
    if output[-1] >= len(full_text.split()):
        output = list(range(start_index, start_index + token_len-1))
    return output
def get_range(item):
    locations = [int(location) for location in item["predictionstring"].split(" ")]
    return (locations[0], locations[-1])
def add_annotation(all_data, start_index, end_index, discourse_type):
    for j in range(start_index, end_index):
        if j == start_index:
            all_data[-1][1][j] = tag_index["B-" + discourse_type]
        else:
            all_data[-1][1][j] = tag_index["I-" + discourse_type]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# begin = time.time()
# last_id = ""
# all_data = []
# ids = []
# match_count = 0
# start_index = 0
# for i in range(len(train)):
#     item = train.iloc[i]
#     identifier = item["id"]
#     discourse_type = item["discourse_type"]
#     if identifier != last_id:
#         last_id = identifier
#         with open(item["file_path"]) as f:
#             content = "".join(f.readlines())
#             tokens = tokenize(content)
#             annotations = [none_index] * len(tokens)
#             all_data.append((tokens, annotations))
#             ids.append(last_id)
#             start_index = 0
#     annotation_range = get_range(item)
#     indices = calc_word_indices(content, int(item["discourse_start"]), int(item["discourse_end"]))
#     if annotation_range[0] == indices[0] and annotation_range[1] == indices[-1]:
#         match_count += 1
#         add_annotation(all_data, annotation_range[0], annotation_range[-1] + 1, discourse_type)
# print(f"Match count: {match_count}, Correct Rate: {match_count / len(train)}")
# print(all_data[0])

"""### Distribution of Word Counts"""

word_counter = defaultdict(int)
for item in all_data:
    for token in item[0]:
        word_counter[token] += 1

word_count = pd.DataFrame({"key": word_counter.keys(), "count": word_counter.values()})
word_count.sort_values(by="count", ascending=False, inplace=True)
word_count.head(30)

plt.figure(figsize=(15, 10))
sns.barplot(x="key", y="count", data=word_count[:30])

word_count.describe()

"""#### Number of words"""

len(word_count)

"""#### Words appearing only once"""

(word_count["count"] <= 1).sum()

"""### Ditrubtion of Sentences Lengths"""

pd.DataFrame({"Sentence Lengths": [len(item[0]) for item in all_data]}).hist()

"""### Vectorization"""

class Vectorizer:

    def __init__(self, vocab_size = None, sequence_length = None, unk_token = "<unk>"):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.unk_token = unk_token

    def fit_transform(self, sentences):
        word_counter = dict()
        for tokens in sentences:
            for token in tokens:
                if token in word_counter:
                    word_counter[token] += 1
                else:
                    word_counter[token] = 1
        word_counter = pd.DataFrame({"key": word_counter.keys(), "count": word_counter.values()})
        word_counter.sort_values(by="count", ascending=False, inplace=True)
        vocab = set(word_counter["key"][0:self.vocab_size-2])
        word_index = dict()
        begin_index = 1
        word_index[self.unk_token] = begin_index
        begin_index += 1
        Xs = []
        for i in range(len(sentences)):
            X = []
            for token in sentences[i]:
                if token not in word_index and token in vocab:
                    word_index[token] = begin_index
                    begin_index += 1
                if token in word_index:
                    X.append(word_index[token])
                else:
                    X.append(word_index[self.unk_token])
                if len(X) == self.sequence_length:
                    break
            Xs.append(X)
        self.word_index = word_index
        self.vocab = vocab
        return Xs

    def transform(self, sentences):
        Xs = []
        for i in range(len(sentences)):
            X = []
            for token in sentences[i]:
                if token in self.word_index:
                    X.append(self.word_index[token])
                else:
                    X.append(self.word_index[self.unk_token])
                if len(X) == self.sequence_length:
                    break
            Xs.append(X)
        return Xs

    def load(self, path):
        with open(path, 'r') as f:
            dic = json.load(f)
            self.vocab_size = dic['vocab_size']
            self.sequence_length = dic['sequence_length']
            self.unk_token = dic['unk_token']
            self.word_index = dic['word_index']

    def save(self, path):
        with open(path, 'w') as f:
            data = json.dumps({
                "vocab_size": self.vocab_size,
                "sequence_length": self.sequence_length,
                "unk_token": self.unk_token,
                "word_index": self.word_index
            })
            f.write(data)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# vectorizer = Vectorizer(vocab_size = vocab_size, sequence_length = sequence_length, unk_token = unk_token)
# token_list = [item[0] for item in all_data]
# if mode == modes[0]:
#     Xs = vectorizer.fit_transform(token_list)
#     vectorizer.save(vectorizer_path)
# else:
#     vectorizer.load(output_dataset_path + vectorizer_path)
#     Xs = vectorizer.transform(token_list)
# for i in range(len(all_data)):
#     item = all_data[i]
#     annotation = item[1]
#     if len(annotation) > sequence_length:
#         annotation = annotation[0:sequence_length]
#     all_data[i] = (Xs[i], annotation)
# train_data, val_data, train_ids, valid_ids = train_test_split(all_data, ids, test_size = 0.1, random_state=42)

"""## Export to files"""

def export_to_file(export_file_path, data):
    with open(export_file_path, "w+") as f:
        for i in range(len(data)):
            X = data[i][0]
            y = data[i][1]
            f.write(
                str(len(X))
                + "\t"
                + "\t".join([str(item) for item in X])
                + "\t"
                + "\t".join([str(item) for item in y])
                + "\n"
            )

export_to_file("train.txt", train_data)
export_to_file("validation.txt", val_data)

"""## Create Tensorflow Dataset"""

def preprocess(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tokens = tf.strings.to_number(tokens, out_type=tf.int64)
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    return tokens, tags
def make_dataset(file_path, batch_size, mode="train"):
    ds = tf.data.TextLineDataset(file_path).map(preprocess)
    if mode == "train":
        ds = ds.shuffle(256)
    ds = ds.padded_batch(batch_size)

    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset("train.txt", batch_size, mode="train")
val_ds = make_dataset("validation.txt", batch_size, mode="valid")

for X, y in train_ds.take(2):
    print(X)
    print(y)

"""## Modeling"""

accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
def accuracy(y_true, y_pred):
    acc = accuracy_metric(y_true, y_pred)
    mask = tf.cast((y_true > 0), dtype=tf.float32)
    acc = acc * mask
    return tf.reduce_sum(acc) / tf.reduce_sum(mask)

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

model = keras.Sequential([
    PositionalEmbedding(sequence_length, vocab_size, embed_size),
    TransformerBlock(embed_size, 4, 32),
    TransformerBlock(embed_size, 4, 32),
    layers.Dropout(0.1),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(len(all_tags), activation="softmax")
])

"""## Training"""

if mode == modes[0]:
    early_stop = keras.callbacks.EarlyStopping(
        min_delta=1e-4,
        monitor="val_loss",
        patience=10
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.3,
        patience=4,
        monitor="val_loss",
        min_lr=1e-7
    )
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss = CustomNonPaddingTokenLoss()
    #loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy])
    callbacks = [early_stop, reduce_lr]

    model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    model.save_weights(model_path)
else:
    model.build(input_shape=(None, None))
    model.load_weights(output_dataset_path+model_path)

model.summary()

keras.utils.plot_model(model, show_shapes=True, show_dtype=True)

"""### Evaluation"""

def evaluate(model, dataset):
    all_true_tag_ids, all_predicted_tag_ids = [], []
    for x, y in dataset:
        output = model.predict(x)
        predictions = np.argmax(output, axis=-1)
        predictions = np.reshape(predictions, [-1])

        true_tag_ids = np.reshape(y, [-1])

        mask = (true_tag_ids != 0) & (true_tag_ids != none_index) & (predictions > 0)
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predicted_tag_ids)

    all_true_tag_ids = np.concatenate(all_true_tag_ids)
    all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)
    cls_report = classification_report(all_true_tag_ids, all_predicted_tag_ids)
    print("Classifiction report:")
    print(cls_report)
    f1 =  f1_score(all_true_tag_ids, all_predicted_tag_ids, average="macro")
    print("F1 score:", f1)

def create_prediction_csv_file(model, dataset, origin_ids, file_path):
    predictions = []
    classes = []
    ids = []
    t = 0
    for item in dataset:
        if len(item) == 2:
            X = item[0]
        else:
            X = item
        y_pred =  np.argmax(model.predict(X), axis=-1)
        for i in range(y_pred.shape[0]):
            last_prediction = None
            indices = []
            identifier = origin_ids[t]
            t += 1
            for j in range(X.shape[1]):
                if last_prediction != index_tag[y_pred[i, j]]:
                    if len(indices) > 0:
                        ids.append(identifier)
                        predictions.append(indices)
                        classes.append(last_prediction)
                        indices = []
                    last_prediction = index_tag[y_pred[i, j]]
                if y_pred[i, j] != pad_index and y_pred[i, j] != none_index:
                    indices.append(j)
                if j == X.shape[1] - 1:
                    if len(indices) > 0:
                        ids.append(identifier)
                        predictions.append(indices)
                        classes.append(last_prediction)
                        indices = []
                if X[i, j] == pad_index:
                    break
    new_ids = []
    new_classes = []
    new_preditions = []
    for i in range(len(ids)):
        merge = False
        if ids[i - 1] == ids[i] and i > 0:
            if len(predictions[i]) <= 3:
                merge = True
                j = new_preditions[-1][-1] + 1
                while j < predictions[i][0]:
                    new_preditions[-1].append(j)
                    j += 1
                new_preditions[-1] = new_preditions[-1] + predictions[i]
            elif abs(predictions[i][0] - new_preditions[-1][-1]) <= 3 and classes[i] == new_classes[-1]:
                merge = True
                j = new_preditions[-1][-1] + 1
                while j < predictions[i][0]:
                    new_preditions[-1].append(j)
                    j += 1
                new_preditions[-1] = new_preditions[-1] + predictions[i]
        if not merge:
            new_ids.append(ids[i])
            new_classes.append(classes[i])
            new_preditions.append(predictions[i])
    df = pd.DataFrame({"id": new_ids, "class": new_classes, "predictionstring": [" ".join([str(element) for element in item]) for item in new_preditions]})
    df.to_csv(file_path, index=False)

evaluate(model, val_ds)

"""## Submission"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# token_list = []
# for i in range(len(submission)):
#     item = submission.iloc[i]
#     identifier = item["id"]
#     with open(item["file_path"]) as f:
#         content = "".join(f.readlines())
#         tokens = tokenize(content)
#         token_list.append(tokens)

def preprocess_test(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tokens = tf.strings.to_number(tokens, out_type=tf.int64)
    return tokens
def make_test_dataset(Xs, file_path, batch_size):
    with open(file_path, "w+") as f:
        for i in range(len(Xs)):
            X = Xs[i]
            f.write(
                str(len(X))
                + "\t"
                + "\t".join([str(item) for item in X])
                + "\n"
            )
    ds = tf.data.TextLineDataset(file_path).map(preprocess_test)
    ds = ds.padded_batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds

# Commented out IPython magic to ensure Python compatibility.
# %%time
# X_test = vectorizer.transform(token_list)
# test_ds = make_test_dataset(X_test, "test.txt", batch_size)
# create_prediction_csv_file(model, test_ds, list(submission["id"]), "submission.csv")
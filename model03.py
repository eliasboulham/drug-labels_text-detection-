import cv2
import numpy as np
import random
from PIL import ImageFont
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
dircm = "model06"
dircd = "data05ls"
dircmr="/content/drive/MyDrive/models/model-r"
o = 5000
TS = ["Vignette", "LABORATIO", "BEKER", "Metronida", "BEKER",
      "TARIF", "DE", "Boite", "160.00 DA", "REF", "LOT", "RER", "FAB", "PPA"]
import os
import numpy as np
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import logging
logging.getLogger('tensorflow').setLevel(logging.WARNING)
batch_size = 200
img_width = 200
img_height = 50
def split_data(images, labels, train_size=0.8, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)
    x_train, y_train= images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid= images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid
def encode_single_sample(img_path, label):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}
def treat(dircd):
    # Path to the data directory
    global train_dataset
    global max_length
    global num_to_char
    global validation_dataset
    data_dir = Path(dircd)
    # Get list of all the images
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    characters1 = set(char for label in labels for char in label)
    characters = sorted(characters1)
    characters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','*','+','-','=','_',
                '$','(',')','@',':','%']
    characters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E',
     'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
     'U', 'V', 'W', 'X', 'Y', 'Z',
     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h','i', 'j', 'k', 'l', 'm', 'n', 'o',
     'p','q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
     '*','+','-','=','_','@','(',')',' ',':','%',',','$']
    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("(:::::::::::::::::)")
    print("Characters present: ", characters)
    print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFf")
    global char_to_num
    max_length = max([len(label) for label in labels])
    char_to_num = layers.StringLookup( vocabulary=list(characters), mask_token=None)
    num_to_char = layers.StringLookup( vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
    print("#########################\n\n")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))
    print("###################3",len(char_to_num.get_vocabulary()))
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        # At test time, just return the computed predictions
        return y_pred
def build_model():
        # Inputs to the model
        input_img = layers.Input(
            shape=(img_width, img_height, 1), name="image", dtype="float32"
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")



        # First conv block
        x = layers.Conv2D(32, (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((img_width // 4), (img_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(
            len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
        )(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = keras.models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = keras.optimizers.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
#model=build_model()
def get_pr(ss):
    model=keras.models.load_model(ss)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    return  prediction_model
def get_mod(ss):
    model=keras.models.load_model(ss)
    return model
def one_sam(ss):
    #validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid[i:i+1], y_valid[i:i+1]))
    validation_dataset = tf.data.Dataset.from_tensor_slices(([ss],["gggggg"]))

    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    sam = validation_dataset
    o = next(iter(sam))
    img = o["image"]
    lab = o["label"]
    return img
def normsam(img):
    cv2.imwrite("world01.png",img)
    #validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid[i:i+1], y_valid[i:i+1]))
    validation_dataset = tf.data.Dataset.from_tensor_slices((["world01.png"],["gggggg"]))

    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    sam = validation_dataset
    o = next(iter(sam))
    img = o["image"]
    lab = o["label"]
    return img
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    print("the shape ",input_len)
    print(" pred Sahpe :",pred.shape)
    print(" max lenght " ,max_length)
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][ :, :max_length]
    # Iterate over the results and get back the text
    print("reult ==  ",results)
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
def test (pr,ss):
    img=one_sam(ss)
    rr = pr.predict(img)
    rrd=decode_batch_predictions(rr)
    print("the result : ",rrd )
    return rrd ;
def test2 (pr,img):
    cv2.imwrite("world01.png", img)

    img=one_sam("world01.png")
    rr = pr.predict(img)
    rrd=decode_batch_predictions(rr)
    print("the result : ",rrd )
    return rrd ;
def start_train():
    model=build_model()
    epochs = 100
    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )
    model.save(dircm)
    print("training has finished successful ")
def resume_train(model,dircmr):

    epochs = 100
    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )
    model.save(dircmr)
    print("training has finished successful ")

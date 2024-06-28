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
# Path to the data directory
"""data_dir = Path("./new/")
images = sorted(list(map(str, list(data_dir.glob("*.png")))))

labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]

characters1 = set(char for label in labels for char in label)
characters = sorted(characters1)
"""




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



#print("Number of images found: ", len(images))
#print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

batch_size = 64
img_width = 200
img_height = 50
max_length = 10
char_to_num = layers.StringLookup( vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup( vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
"""x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))
"""
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
"""batch = next(iter(train_dataset.take(1)))

batch = next(iter(m.validation_dataset.take(1)))

images = batch["image"][0]
print("LLLLLLLLLLL",images)"""
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
print("#########################\n\n")
"""train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
     encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size) .prefetch(buffer_size=tf.data.AUTOTUNE))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
     encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))"""
def my_fitting(cc,x=50,y=200):
    a,b,_=cc.shape
    r=a/x
    w=b/r
    if(w<y):
        w=round(w)
        cc = cv2.resize(cc, (w, x))
        cc = cv2.copyMakeBorder(cc, 0, 0, 0, y-w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        cc = cv2.resize(cc, (y, x))

    return cc



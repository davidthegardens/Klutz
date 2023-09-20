import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
from keras import mixed_precision
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import polars as pl
from sklearn.utils.class_weight import compute_sample_weight
from CustomCallback import CustomSaver
from sklearn.metrics import classification_report
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.preprocessing import LabelBinarizer

# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
Input=tf.keras.layers.Input
Model=tf.keras.models.Model

df = pl.read_parquet('data_with_tokens.parquet')

X_train_seq, X_test_seq, y_train, y_test = train_test_split(df['Encoded'], df['category'],test_size=0.3, random_state=2)

# Convert your data into a tf.data.Dataset
BUFFER_SIZE = 10000  # Adjust this value as needed
BATCH_SIZE = 64  # Adjust this value as needed

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_seq, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)



encoder = LabelBinarizer()
y_train_onehot = encoder.fit_transform(y_train)
y_test_onehot = encoder.transform(y_test)

# Assuming you have defined create_generator, create_discriminator, and create_gan functions
def create_generator(max_length):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=30000, output_dim=100, input_length=max_length))
    model.add(tf.keras.layers.LSTM(100, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=100, activation='softmax'))
    return model

def create_discriminator(max_length, y_train_onehot):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=100, output_dim=100, input_length=max_length))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dense(y_train_onehot.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],jit_compile=True)
    return model

def create_gan(discriminator, generator, max_length):
    discriminator.trainable = False
    gan_input = Input(shape=(max_length,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='categorical_crossentropy', optimizer='adam',jit_compile=True)
    return gan

# Training loop
def train_gan(epochs, batch_size, max_length, train_dataset, y_train_onehot, gan, discriminator):
    for epoch in range(epochs):
        for X_batch, y_batch in train_dataset:
            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X_batch, y_batch)

            # Train generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, size=(batch_size, max_length))
            fake_labels = np.random.randint(0, y_train_onehot.shape[1], size=batch_size)
            gan.train_on_batch(noise, fake_labels)

# In the main function, replace X_train_seq with train_dataset
if __name__ == "__main__":
    # Create new instances of the generator and discriminator
    max_length = max(len(seq.split()) for seq in X_train_seq)
    generator = create_generator(max_length)
    discriminator = create_discriminator(max_length, y_train_onehot)

    # Create the GAN
    gan = create_gan(discriminator, generator, max_length)

    # Train the GAN
    train_gan(300, 500, max_length, train_dataset, y_train_onehot, gan, discriminator)
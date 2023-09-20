import numpy as np
import tensorflow as tf
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import polars as pl
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from CustomCallback import CustomSaver
from sklearn.metrics import classification_report
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
df = pl.read_parquet('data_with_tokens.parquet')

X_train_seq, X_test_seq, y_train, y_test = train_test_split(df['Encoded'], df['category'],test_size=0.3, random_state=2)
# Create a BPE model

def bytePair_gen():
    tokenizer = Tokenizer(BPE())

    # Initialize a trainer
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer()

    # Train the tokenizer
    tokenizer.train_from_iterator(X_train, trainer)

    # Save the tokenizer
    tokenizer.save("bpe_tokenizer.json")

def encode():
    # Load the tokenizer
    tokenizer = Tokenizer.from_file("bpe_tokenizer.json")
    X_train_encoded = [tokenizer.encode(text).ids for text in X_train]
    X_test_encoded = [tokenizer.encode(text).ids for text in X_test]
    return X_train_encoded,X_test_encoded,tokenizer

#bytePair_gen()
#X_train_encoded,X_test_encoded,tokenizer=encode()

Input=tf.keras.layers.Input
Model=tf.keras.models.Model





pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
max_length = max(len(seq) for seq in X_train_seq)
X_train_seq = pad_sequences(X_train_seq, maxlen=max_length)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_length)

encoder = LabelEncoder()
y_train_int = encoder.fit_transform(y_train)
y_test_int = encoder.transform(y_test)

to_categorical=tf.keras.utils.to_categorical
y_train_onehot = to_categorical(y_train_int)
y_test_onehot = to_categorical(y_test_int)

# Define the generator
def create_generator(max_length):
    print(max_length)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=30000, output_dim=100, input_length=max_length))
    model.add(tf.keras.layers.LSTM(100, return_sequences=False))
    model.add(tf.keras.layers.Dense(units=100, activation='softmax'))
    return model

# Define the discriminator
def create_discriminator(max_length, y_train_onehot):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=100, output_dim=100, input_length=max_length))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dense(y_train_onehot.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create GAN
def create_gan(discriminator, generator, max_length):
    discriminator.trainable = False
    gan_input = Input(shape=(max_length,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='categorical_crossentropy', optimizer='adam')
    return gan

# Training loop
def train_gan(epochs, batch_size, max_length, X_train_seq, y_train_onehot, gan, discriminator):
    for epoch in range(epochs):
        # Train discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(X_train_seq, y_train_onehot)

        # Train generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, size=(batch_size, max_length))
        fake_labels = np.random.randint(0, y_train_onehot.shape[1], size=batch_size)
        gan.train_on_batch(noise, fake_labels)

if __name__ == "__main__":
    # Define your parameters here
    # max_length = # Your value here
    # tokenizer = # Your value here
    # y_train_onehot = # Your value here
    # X_train_seq = # Your value here
    # epochs = # Your value here
    # batch_size = # Your value here

    # Create new instances of the generator and discriminator
    generator = create_generator(max_length)
    discriminator = create_discriminator(max_length, y_train_onehot)

    # Create the GAN
    gan = create_gan(discriminator, generator, max_length)

    # Train the GAN
    train_gan(300, 500,max_length, X_train_seq, y_train_onehot, gan, discriminator)
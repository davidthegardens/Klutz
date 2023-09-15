import tensorflow as tf
#from keras.preprocessing.text import Tokenizer
# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense
# from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import polars as pl
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

df = pl.read_parquet('ready_training_data.parquet')

X_train, X_test, y_train, y_test = train_test_split(df['merged_opcodes'], df['category'], random_state=1)

# Assuming `X_train`, `X_test`, `y_train`, `y_test` are your data

tokenizer =tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)



# Pad sequences to the same length
pad_sequences=tf.keras.preprocessing.sequence.pad_sequences
max_length = max(len(seq) for seq in X_train_seq)
X_train_seq = pad_sequences(X_train_seq, maxlen=max_length)
X_test_seq = pad_sequences(X_test_seq, maxlen=max_length)

# Convert categories to integers
encoder = LabelEncoder()
y_train_int = encoder.fit_transform(y_train)
y_test_int = encoder.transform(y_test)

# Convert integers to one-hot vectors
to_categorical=tf.keras.utils.to_categorical
y_train_onehot = to_categorical(y_train_int)
y_test_onehot = to_categorical(y_test_int)


# Create LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_length))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(y_train_onehot.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
class_weights = compute_sample_weight(class_weight='balanced', y=y_train_int)
class_weight_dict = dict(enumerate(class_weights))
model.fit(X_train_seq, y_train_onehot, epochs=10, class_weight=class_weight_dict)

# Make predictions
predictions = model.predict_classes(X_test_seq)
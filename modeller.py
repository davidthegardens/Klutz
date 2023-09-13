from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import polars as pl

# Assuming you have a DataFrame `df` with 'text' and 'category' columns
df = pd.read_csv('your_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], random_state=1)

# Create a pipeline that first transforms the text data into a bag-of-words
# feature matrix, then applies SMOTE for minority oversampling, and finally
# trains a Naive Bayes classifier
pipeline = make_pipeline(
    CountVectorizer(),
    SMOTE(random_state=42),
    MultinomialNB()
)

# Train the model
pipeline.fit(X_train, y_train)

# Now you can use the trained model to predict the category of a new text
def predict_category(text):
    return pipeline.predict([text])[0]

# Test the function with a new text
print(predict_category("Your long string here"))
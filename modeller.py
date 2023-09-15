from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from joblib import dump
import pandas as pd
import polars as pl
import fake_code_generator
from sklearn.metrics import confusion_matrix,classification_report

# Assuming you have a DataFrame `df` with 'text' and 'category' columns
# df = pl.read_parquet('datafiles/data_with_code655.parquet')
# df=df.fill_nan(None)
# df=df.drop_nulls()
# df_chars=pl.read_parquet('opcode_to_char.parquet')
# charlist=df_chars['characters'].to_list()
# characters="".join(charlist)

def addFake(df,characters,entries):
    col=[]
    pdf=df.to_pandas()
    for i in range(entries):
        col.append(fake_code_generator.generateCode(pdf,characters))
        print(f"{i+1} of {entries} false entries generated.")
    new_df=pl.DataFrame({'name':None, 'category':"Artificial", 'address':None, 'family':None, 'bytecodes':None, 'opcodes_chars':None, 'merged_opcodes':col})
    new_df=new_df.with_columns(pl.col('name').cast(pl.Utf8),pl.col('category').cast(pl.Utf8),pl.col('address').cast(pl.Utf8),pl.col("family").cast(pl.List(pl.Utf8)),pl.col("bytecodes").cast(pl.List(pl.Utf8)),pl.col("opcodes_chars").cast(pl.List(pl.Utf8)))
    new_df = pl.concat([df, new_df], rechunk=True)
    return new_df

# df=addFake(df,characters,200000)
# df.write_parquet('data_with_fake_code.parquet')
# Split the data into training and testing sets


# counts=y_train.value_counts()
# print(len(counts['category'].to_list()))
# counts=counts.filter(pl.col('counts')>5)
# categories=counts['category'].to_list()
# df=df.filter(pl.col('category').is_in(categories))
# df.write_parquet('ready_training_data.parquet')


# Create a pipeline that first transforms the text data into a bag-of-words
# feature matrix, then applies SMOTE for minority oversampling, and finally
# trains a Naive Bayes classifier
def loadTrain():
    df = pl.read_parquet('ready_training_data.parquet')
    X_train, X_test, y_train, y_test = train_test_split(df['merged_opcodes'], df['category'], random_state=1)

    pipeline = make_pipeline_imb(
        CountVectorizer(),
        SMOTE(random_state=42,k_neighbors=5),
        MultinomialNB()
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the model to a file
    dump(pipeline, 'text_classification_model.joblib')
    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)

loadTrain()
# from joblib import load
# pipeline = load('text_classification_model.joblib')
# predictions = pipeline.predict(X_test)
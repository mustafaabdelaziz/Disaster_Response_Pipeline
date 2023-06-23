import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """ Loading data from SQL Database

    Args:
        database_filepath: The file path for the SQL DataBase.

    Returns:
        X: A dataFrame contains The messages.
        Y: A DataFrame contains the Categories.
        category_names: A list of Category names.
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[3:36]
    return X, Y, category_names


def tokenize(text):
    """Transforming the message Text into a machine learning usable form.

    Args: 
        text: The message sent by the user.

    Returns:
        lemmated: Transformed and cleaned list of words. 
    """
    # Normalization
    text = re.sub(r"\W", " ", text.lower())

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization and stop words removel
    lemmatizer = WordNetLemmatizer()
    lemmated = [lemmatizer.lemmatize(w).strip(
    ) for w in tokens if w not in stopwords.words('english')]

    return lemmated


def build_model():
    """Building the Machine learning Pipeline to predict the categories of text.

    Args:
        None

    Returns:
        cv: Optimized machine learning pipeline.
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tf-idf", TfidfTransformer()),
        ("clf", KNeighborsClassifier())
    ])
    parameters = {
        #         'vect__ngram_range': ((1, 1), (1, 2)),
        # the optemal n_neighbors number is 10
        'clf__n_neighbors': [5, 10, 15, 30, 50],
        'clf__leaf_size': [20, 30, 50]  # the optemal leaf_size is 20
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluating the Machine Learning model.

    Args:
        model: The machine learning model used to classefy text categories.
        X_test: Messages used as Testing Dataset.
        Y_test: The real categories of the testing messages.
        category_names: The names of the categories.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    labels = category_names

    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """Saving Classification model.

    Args:
        model: The machine learning model used to classefy text categories.
        model_filepath: The file bath to save the model at.

    Returns:
        None.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

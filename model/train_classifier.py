"""
Disaster Response Classification Model Trainer

This script loads disaster response data from a SQLite database, preprocesses text,
trains a multi-output classification model using a machine learning pipeline, and saves
the trained model as a pickle file.

Usage:
    python train_classifier.py <database_filepath> <model_filepath>

Arguments:
    database_filepath : str : Path to SQLite database file.
    model_filepath    : str : Path to save the trained model as a pickle file.
"""

import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target labels.

    Args:
        database_filepath (str): Path to SQLite database.

    Returns:
        X (numpy.ndarray): Feature data (messages).
        Y (numpy.ndarray): Target labels (categories).
        category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM DisasterMessages', engine)
    
    X = df['message'].values  # Messages as input features
    Y = df.iloc[:, 4:].values  # Category labels
    category_names = df.columns[4:]  # Category names

    return X, Y, category_names

def tokenize(text):
    """
    Normalize, tokenize, and lemmatize text.

    Args:
        text (str): Input text.

    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,()]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline with GridSearchCV for hyperparameter tuning.

    Returns:
        cv (GridSearchCV): Grid search model with a pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [10, 50],  
        'clf__estimator__min_samples_split': [2, 3],  
        'clf__estimator__max_depth': [None, 10]  
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,
                       cv=2, n_jobs=-1, verbose=2)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance and print classification reports.

    Args:
        model (sklearn model): Trained model.
        X_test (numpy.ndarray): Test feature data.
        Y_test (numpy.ndarray): True labels for test data.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    print("\nClassification Report:\n")
    accuracy_list = []
    
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
        accuracy_list.append(acc)
        print("-" * 60)
    
    overall_accuracy = np.mean(accuracy_list)
    print(f"\nOverall Model Accuracy: {overall_accuracy:.4f}")

def save_model(model, model_filepath):
    """
    Save trained model to a pickle file.

    Args:
        model (sklearn model): Trained model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    """
    Main function to execute the training pipeline.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument.\n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()

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

# Function to load data from SQLite database
def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('df', engine)

    X = df['message'].values  # Message column as input
    Y = df.iloc[:, 4:].values  # All categories as output
    category_names = df.columns[4:]  # Names of categories

    return X, Y, category_names

# Function to clean and tokenize text
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens

# Function to build a machine learning pipeline
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to count vectors
        ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classification
    ])

    # Hyperparameter tuning using GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100],  # Number of trees in the forest
        'clf__estimator__min_samples_split': [2, 4]  # Minimum samples per split
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    return cv

# Function to evaluate the model performance
def evaluate_model(model, X_test, Y_test, category_names):
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

# Function to save the trained model as a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

# Main execution function
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
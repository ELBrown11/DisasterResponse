import sys
import pandas as pd
import sqlite3
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets, merge them on 'id'."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """Cleans the merged dataset by splitting categories, renaming columns, and removing duplicates."""
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    category_colnames = categories.iloc[0].str[:-2].tolist()
    categories.columns = category_colnames
    
    # Convert category values to numeric (vectorized approach)
    categories = categories.applymap(lambda x: int(x[-1]))  

    # Drop rows where any category column contains values other than 0 or 1
    categories = categories[(categories == 0) | (categories == 1)].dropna()
    
    # Drop original categories column & merge cleaned categories back
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
  

def save_data(df, database_filename):
    """Saves cleaned data to an SQLite database."""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')  # 'if_exists' prevents duplication
    # Confirm that data was saved
    with sqlite3.connect(database_filename) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM DisasterMessages;")
        count = cursor.fetchone()[0]
        print(f"Successfully saved {count} rows to DisasterMessages")

def main():
    """Main function to execute data loading, cleaning, and saving."""
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()

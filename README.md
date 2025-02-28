# Machine Learning for Disaster Response
This project leverages ETL (Extract, Transform, Load) and Machine Learning pipelines to process and classify disaster-related messages. The goal is to help emergency responders quickly identify and prioritize messages during crises.

The results of these pipelines are displayed through an interactive Flask web application, which includes:
- A bar chart showing the count of message types across different genres.
- A pie chart displaying the percentage of disaster-related vs. non-disaster-related messages.
- A text classifier that predicts the topic of an input message.
- A word cloud highlighting the most common words in disaster-related messages.
## Project Outline
* ETL Pipeline
  * wrote a python script to preprocess and clean data
  * loads datasets
  * merges the datasets
  * cleans data
  * stores data into SQLite database     
* Machine Learning Pipeline
  * loads data stored via ETL pipeline from database
  * split data into training and test
  * building text and machine learning pipeline
  * use GridSearchCV() model 
* Flask Web App



# Languages & Libraries
* languages: Python, SQL
* libraries: pandas, matplotlib, numpy, sqlite, sqlalchemy,nltk, sklearn, re

# Tools & Software
* Google Colab
* Google Documents
* VS Code

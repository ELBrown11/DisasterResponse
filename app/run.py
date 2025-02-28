import json
import plotly
import pandas as pd
import base64
from io import BytesIO
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """Tokenize and lemmatize input text."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM DisasterMessages', engine)

# Load model
model = joblib.load("../models/classifier_model.pkl")


@app.route('/')
@app.route('/index')
def index():
    """Index page with visualizations."""
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Create visuals (Bar chart)
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # Create visuals (Pie chart)
    # Pie chart based on message classification (related vs. not related)
    related_counts = df['related'].value_counts()
    related_labels = ['Not Related', 'Related']
    graphs.append(
        {
            'data': [
                Pie(
                    labels=related_labels,
                    values=related_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Relation to Disaster'
            }
        }
    )

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with Plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


@app.route('/go')
def go():
    """Handles user query and displays model results."""
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Generate word cloud image
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['message']))
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_img = base64.b64encode(img.getvalue()).decode()

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        wordcloud_img=wordcloud_img
    )


def main():
    """Run the Flask app."""
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()

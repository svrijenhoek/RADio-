import pandas as pd
import ast

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

articles = pd.DataFrame()
behaviors = pd.DataFrame()
predictions = pd.DataFrame()


"""
Behavior file preprocessing. This includes:
- Transforming string-columns to lists
- Adding an extra column that contains the news articles in the impression without the click indication
- Sampling
"""


def remove_suffix(lst):
    return [item.split("-")[0] for item in lst]


def process_behavior(location, sample_size):
    global behaviors
    behaviors = pd.read_csv(location, sep='\t',
                            names=['impr_index', 'userid', 'date', 'history', 'items'], index_col=0)

    behaviors['history'] = behaviors['history'].str.split()
    behaviors['items'] = behaviors['items'].str.split()
    behaviors['impression'] = behaviors['items'].apply(remove_suffix)

    if sample_size > 0:
        behaviors = behaviors.sample(n=sample_size)

    return behaviors


"""
Article preprocessing. This includes:
- Sentiment analysis
- Adding an extra column that contains the people mentioned in the title entities and subtitle entities columns
"""

# Initialize the VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()


# Define a function to calculate sentiment scores
def get_sentiment_score(text):
    sentiment = sid.polarity_scores(text)
    return abs(sentiment['compound'])


# Code for extracting the people mentioned in the title and subtitle
def extract_people(row):
    try:
        entries = ast.literal_eval(row)
        labels = [entry['Label'] for entry in entries if entry['Type'] == 'P']
        return labels
    except ValueError:
        return []


def select(lst, feature):
    try:
        return list(articles.loc[lst][feature])
    except KeyError:
        return []


def process_articles(location):
    global articles
    articles = pd.read_csv(location, sep='\t',
                           names=['article_id', 'category', 'subcategory', 'title', 'subtitle', 'url', 'entities_title',
                                  'entities_subtitle'], index_col=0)

    # Apply the function to the 'text' column to create a new 'sentiment_score' column
    articles['absolute_sentiment_score'] = articles['title'].apply(get_sentiment_score)

    # Apply function to the column and create new column with the extracted labels
    articles['persons'] = articles['entities_title'].apply(extract_people) + articles['entities_subtitle'].apply(
        extract_people)
    return articles


"""
Preparing the prediction files. This means:
- Filters the predictions to only include the ones that are also in the behaviors file (in case of sampling)
- Adding an extra column that has the articles in the impression ordered by their predicted rank
- Adding an extra column for the date of the prediction
"""


def order(row):
    prediction = row['pred_rank']
    behavior_row = behaviors.loc[row.name]
    # match the predicted ordering from pred_rank with the right article id as it is specified in
    # the behavior file impression
    ordered = [behavior_row.impression[prediction[i] - 1] for i in range(len(prediction))]
    return ordered


def process_predictions(location):
    global predictions
    predictions = pd.read_json(location, lines=True).set_index('impr_index')
    # only include the predictions that are in the behavior file sample
    predictions = predictions.loc[behaviors.index]
    # add column where the recommendation is the ranked list of article ids
    predictions['recommendation'] = predictions.apply(order, axis=1)
    # add date of prediction
    predictions = predictions.join(behaviors['date'])
    return predictions


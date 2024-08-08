import pandas as pd
import ast
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

articles = pd.DataFrame()
behaviors = pd.DataFrame()
predictions = pd.DataFrame()


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


def make_list(lst, feature):
    """
    Function that helps extract the relevant features for a recommendation. Returns for each impr_index in the list the
    corresponding feature, in the same order as in the list.

    Example for feature 'category':
        articles: [{'impr_index': 'N1', 'category': 'news'},{'impr_index': 'N2', 'category': 'sport'},{'impr_index': 'N3', 'category': 'entertainment'}]
        lst : ['N2','N3','N1']
        output: ['sport','entertainment','news']
    """
    try:
        result = list(articles.loc[lst][feature])
        return result
    except KeyError:
        return []



"""
Preparing the prediction files. This means:
- Filters the predictions to only include the ones that are also in the behaviors file (in case of sampling)
- Adding an extra column that has the articles in the impression ordered by their predicted rank
- Adding an extra column for the date of the prediction
"""


def order(row, recommendation_length):
    prediction = row['pred_rank']
    behavior_row = behaviors.loc[row['impr_index']]
    # match the predicted ordering from pred_rank with the right article id as it is specified in
    # the behavior file impression
    ordered = [behavior_row.impression[prediction[i] - 1] for i in range(min(len(prediction), recommendation_length))]
    return ordered


def process_recommendations(location, sample_size):
    pred = pd.read_json(location)

    columns = list(pred.columns)
    not_algorithms = ['impr_index', 'userid', 'date', 'history']
    algorithms = [ele for ele in columns if ele not in not_algorithms]

    if sample_size > 0:
        unique_users = pred.userid.unique()
        selected_users = np.random.choice(unique_users, size=sample_size, replace=False)
        pred = pred[pred['userid'].isin(selected_users)]
    return algorithms, pred

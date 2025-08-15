import snscrape.modules.twitter as sntwitter
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

def fetch_twitter_sentiment(keyword="Bitcoin", days=7):
    analyzer = SentimentIntensityAnalyzer()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    tweets_list = []

    query = f'{keyword} since:{start_date.date()} until:{end_date.date()} lang:en'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i > 500:  # limit for speed
            break
        score = analyzer.polarity_scores(tweet.content)['compound']
        tweets_list.append([tweet.date.date(), score])

    df = pd.DataFrame(tweets_list, columns=['Date', 'Sentiment'])
    sentiment_daily = df.groupby('Date')['Sentiment'].mean().reset_index()
    return sentiment_daily

import pandas as pd
from util.openai_service import analyze_sentiment

def process_sentiment(df):
    df = df.copy()

    df['sentiment'] = df.apply(lambda row: analyze_sentiment(row['text']) if pd.isnull(row.get('sentiment')) else row['sentiment'], axis=1)

    return df


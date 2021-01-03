from imports import *

def prepare_dataset():
    # Load dataset
    df = pd.read_csv('wiki_movie_plots_deduped.csv')

    # Remove movies with 'unknown' and '-' genres
    df = df[df['Genre'] != 'unknown']
    df = df[df['Genre'] != '-']

    # Remove parts in brackets
    df['Genre'] = df['Genre'].str.replace("[\(\[].*?[\)\]]", "")

    # Handle genres (split by / - , ; etc.)
    new_df = pd.DataFrame(df['Genre'].str.split('/').tolist(), index=df['Title']).stack()
    new_df = new_df.reset_index([0, 'Title'])
    new_df.columns = ['Title', 'Genre']
    df = new_df

    new_df = pd.DataFrame(df['Genre'].str.split(' - ').tolist(), index=df['Title']).stack()
    new_df = new_df.reset_index([0, 'Title'])
    new_df.columns = ['Title', 'Genre']
    df = new_df

    new_df = pd.DataFrame(df['Genre'].str.split(',').tolist(), index=df['Title']).stack()
    new_df = new_df.reset_index([0, 'Title'])
    new_df.columns = ['Title', 'Genre']
    df = new_df

    new_df = pd.DataFrame(df['Genre'].str.split(';').tolist(), index=df['Title']).stack()
    new_df = new_df.reset_index([0, 'Title'])
    new_df.columns = ['Title', 'Genre']
    df = new_df

    df['Genre'] = df['Genre'].str.strip() # trim strings after split

    # Drop movies with no genre
    df = df[df['Genre'] != '']

    return df
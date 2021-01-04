from imports import *

def prepare_dataset():
    # Load dataset
    df = pd.read_csv('wiki_movie_plots_deduped.csv')
    print("Original dataset shape: ", df.shape)

    # Remove movies with 'unknown' and '-' genres
    df = df[df['Genre'] != 'unknown']
    df = df[df['Genre'] != '-']

    # Remove parts in brackets (some genres contain [num] or [not in citation given], others contain irrelevant details in brackets)
    df['Genre'] = df['Genre'].str.replace("[\(\[].*?[\)\]]", "")

    # Split multiple genres by / - , ; & etc.
    df['Genre'] = df['Genre'].str.split("/")
    df = df.explode('Genre').reset_index(drop=True)

    df['Genre'] = df['Genre'].str.split(" - ")
    df = df.explode('Genre').reset_index(drop=True)

    df['Genre'] = df['Genre'].str.split("&")
    df = df.explode('Genre').reset_index(drop=True)

    df['Genre'] = df['Genre'].str.split(",") # conflict with genre that contains '4,000'
    df = df.explode('Genre').reset_index(drop=True)

    df['Genre'] = df['Genre'].str.split(";")
    df = df.explode('Genre').reset_index(drop=True)

    # Trim strings after split
    df['Genre'] = df['Genre'].str.strip()

    # Drop movies with no genre
    df = df[df['Genre'] != '']

    print("Final dataset shape: ", df.shape)

    return df
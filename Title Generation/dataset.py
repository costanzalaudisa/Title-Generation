from imports import *


def build_vocab(df):
    vocab = Counter()
    token_list = []
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', punctuation)
    for index, row in df.iterrows():
        plot = row['Plot']
        # split the plot into tokens by white space
        tokens = plot.split()
        #remove punctuation from each token
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        token_list.append(tokens)
    for tokens in token_list:
        vocab.update(tokens)
    return vocab

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

def write_genre_csv():
    # Define and sort genre list
    genres = df['Genre'].unique()
    sorted_genres = sorted(genres)
    print("Length of genre list:", len(sorted_genres), "  |   Genre list: ", sorted_genres)

    action_list = []

    substring_list = ["action", "sci", "dram", "medy", "rom", "horr", "thrill", "crime", "west", "adventure", "music", "fant"]
    substrings = {
        "action"            : ["action"],
        "science-fiction"   : ["sci"],
        "drama"             : ["dram"],
        "comedy"            : ["medy"],
        "romance"           : ["rom"],
        "horror"            : ["horror"],
        "thriller"          : ["thriller"],
        "crime"             : ["crime"],
        "western"           : ["west"],
        "adventure"         : ["adventure"],
        "musical"           : ["music"],
        "fantasy"           : ["fant"]
        }

    with open('modified_ds.csv', 'w', encoding='utf-8') as csvfile:

        csvfile.write("Index, Title, Genre Vec \n")

        for i, item in df.iterrows():
            has_genre = False;
            genre_vec = [0,0,0,0,0,0,0,0,0,0,0,0]
            for index in range(len(substring_list)):
                if substring_list[index] in item['Genre'].lower():
                    genre_vec[index] = 1
                    has_genre = True;
            if (has_genre):
                #print((item['Title'] + "," + str(genre_vec) + "\r\n" ))
                csvfile.write( str(i) + ',' + item['Title'] + "," + str(genre_vec) + "\n")
from imports import *


def build_vocab(df):
    vocab = Counter()
    token_list = []
    stop_words = set(stopwords.words('english'))

    table = str.maketrans('', '', punctuation)
    for index, row in df.iterrows():
        plot = row['Plot']
        # Remove the apostrophe by splitting, else it will be simply removed and create new words
        plot = plot.replace("'"," ")
        # split the plot into tokens by white space
        tokens = plot.split()
        #remove punctuation from each token
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word.lower() for word in tokens if word.isalpha()]
        # filter out stop words
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]
        token_list.append(tokens)

    for tokens in token_list:
        vocab.update(tokens)
    vocab = [k for k,c in vocab.items() if c >= 10]
    data = '\n'.join(vocab)
    # open file
    file = open("vocab.txt", 'w', encoding='utf-8')
    # write text
    file.write(data)
    # close file
    file.close()
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

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def write_genre_csv(df):

    # Load the vocab for the bag-of-words representation of the plot
    vocab = load_doc("vocab.txt")
    vocab = vocab.split()
    vocab = set(vocab)
    tokenizer = Tokenizer()
    table = str.maketrans('', '', punctuation)
    stop_words = set(stopwords.words('english'))
    token_list = list()
    tokenstr_list = list()

    # Define and sort genre list
    genres = df['Genre'].unique()
    sorted_genres = sorted(genres)
    # print("Length of genre list:", len(sorted_genres), "  |   Genre list: ", sorted_genres)

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

        csvfile.write("Index,Title,Genre Vec\n")

        for i, item in df.iterrows():
            has_genre = False;
            has_plot = False;

            genre_vec = [0,0,0,0,0,0,0,0,0,0,0,0]

            # Tokenize the plot and extract the words that are part of the vocab

            plot = item['Plot']
            plot = plot.replace("'"," ")
            # split the plot into tokens by white space
            tokens = plot.split()
            #remove punctuation from each token
            tokens = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            tokens = [word.lower() for word in tokens if word.isalpha()]
            # filter out stop words
            tokens = [w for w in tokens if not w in stop_words]
            # filter out short tokens
            tokens = [word for word in tokens if len(word) > 1]
            tokens = [w for w in tokens if w in vocab]
            tokenstr = ' '.join(tokens)

            token_list.append(tokens)
            tokenstr_list.append(tokenstr)

            # Go through the substrings and check which genres apply fo the current row
            genre = item['Genre'].lower()
            for index in range(len(substring_list)):
                if substring_list[index] in genre:
                    genre_vec[index] = 1
                    has_genre = True;

            # If the row has a genre and a plot, write it into the new csv file
            if (has_genre and has_plot):
                csvfile.write( str(i) + ',' + item['Title'] + "," + str(genre_vec) + "\n")

        CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english')
        Count_data = CountVec.fit_transform(tokenstr_list)

        cv_dataframe = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
        print(cv_dataframe)
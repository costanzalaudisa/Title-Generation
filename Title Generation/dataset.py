from imports import *


def buildVocab(df):
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

def prepareDataset():
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

def getGenreVectors(df):

    # Defined genre list
    genre_list = [
    "action", "science-fiction", "drama", "comedy", "romance",
    "horror", "thriller", "crime", "western", "fantasy", "adventure", "music"]

    genre_vec_list = []


    for i, item in df.iterrows():

    # Go through the substrings and check which genres apply fo the current row
        genre_vec = [0,0,0,0,0,0,0,0,0,0,0,0]
        genre = item['Genre'].split()

        for index in range(len(genre_list)):
            if genre_list[index] in genre:
                genre_vec[index] = 1

        genre_vec_list.append(genre_vec)

    print(len(genre_vec_list))
    return genre_vec_list
    


def getPlotVectors(df):
    # Not tested yet!
    # Load the vocab for the bag-of-words representation of the plot

    vocab = load_doc("vocab.txt")
    vocab = vocab.split()
    vocab = set(vocab)
    tokenizer = Tokenizer()
    table = str.maketrans('', '', punctuation)
    stop_words = set(stopwords.words('english'))
    token_list = list()
    tokenstr_list = list()

    for i, item in df.iterrows():

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

    CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                        stop_words='english')
    Count_data = CountVec.fit_transform(tokenstr_list)

    plot_frame = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    print(plot_frame)
    return plot_frame

def writeCleanedCsv(df):

    # Define and sort genre list
    substrings = {
        "action"    : "action",
        "sci"       : "science-fiction",
        "dram"      : "drama",
        "medy"      : "comedy",
        "rom"       : "romance",
        "hor"       : "horror",
        "thrill"    : "thriller",
        "crim"      : "crime",
        "west"      : "western",
        "fant"      : "fantasy",
        "adv"       : "adventure",
        "mus"       : "music"
        }

    with open('modified_ds.csv', 'w', encoding='utf-8') as csvfile:

        csvfile.write("Title;Genre;Plot\n")

        for i, item in df.iterrows():
            has_genre = False;
            has_plot = False;
            has_title = True;

            plot = item['Plot']
            title = item['Title']
            genre = item['Genre']

            # Check if the entry has a valid title
            # if !valid(title): continue
            title = title.replace('"', '')

            # Extract the predefined genres and write it with a consistent format into the new csv

            genre = genre.lower()
            new_genre = []
            for key in substrings:
                if key in genre:
                    new_genre.append(substrings[key])
                    has_genre = True;

            # Check if the plot is valid: Todo i guess?
            if (plot != ''):
                has_plot = True
                plot = plot.replace("\r\n", " ")
                plot = plot.replace("\n", " ")
                plot = plot.strip('\"')
                plot = plot.replace(";", ",")

            # Check if all three are valid
            if (has_genre and has_plot and has_title):
                title = str("\"" + title + "\"")
                new_genre = str("\"" + " ".join(new_genre) + "\"")
                plot = str("\"" + plot + "\"")

                csvfile.write(title + ";" + str(new_genre) + ";" + plot + "\n")

    df = pd.read_csv('modified_ds.csv', sep=';')
    print("Modified dataset shape: ", df.shape)

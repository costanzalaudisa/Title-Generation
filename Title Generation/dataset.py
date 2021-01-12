from imports import *


def getTitleVectors(df):
    # Get list of unique characters
    vocab = []
    for title in df['Title']:
        for char in list(set(title)):
            vocab.append(char) if char not in vocab else vocab
    vocab = sorted(vocab)
    print(len(vocab), "unique characters\n", vocab)

    ids_from_chars = StringLookup(vocabulary=list(vocab))
    chars_from_ids = StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

    title_list = []
    for i, item in df.iterrows():
        title = item['Title']
        chars = tf.strings.unicode_split(title, input_encoding='UTF-8')
        ids = ids_from_chars(chars)
        title_list.append(ids)
    print("Title list length:", len(title_list))


def buildVocab(df):
    vocab = Counter()
    token_list = []
    stop_words = set(stopwords.words('english'))

    table = str.maketrans('', '', punctuation)
    for index, row in df.iterrows():
        plot = row['Plot']
        # Remove the apostrophe by splitting, else it will be simply removed and create new words
        plot = plot.replace("'"," ")
        plot = plot.strip()
        # Remove text in brackets
        plot = plot.replace('\([\s\S]*\)', "")
        plot = plot.replace('\[[\s\S]*\]', "")
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
    print(vocab)
    data = '\n'.join(vocab)
    # open file
    file = open("vocab.txt", 'w', encoding='utf-8')
    # write text
    file.write(data)
    # close file
    file.close()
    return vocab


def splitGenres(df): # not in use at the moment
    # Load dataset
    print("Original dataset shape: ", df.shape)

    # Remove movies with 'unknown', '-' and empty genres
    df = df[df['Genre'] != 'unknown']
    df = df[df['Genre'] != '-']
    df = df[df['Genre'] != '']

    # Remove parts in brackets (some genres contain [num] or [not in citation given], others contain irrelevant details in brackets)
    df['Genre'] = df['Genre'].str.replace("[\(\[].*?[\)\]]", "")

    # Split multiple genres by / - , ; & etc.
    df['Genre'] = df['Genre'].str.split("/")
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip() # trim after split

    df['Genre'] = df['Genre'].str.split(" - ")
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip() # trim after split

    df['Genre'] = df['Genre'].str.split("&")
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip() # trim after split

    df['Genre'] = df['Genre'].str.split(",") # conflict with genre that contains '4,000'
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip() # trim after split

    df['Genre'] = df['Genre'].str.split(";")
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].str.strip() # trim after split

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

    print(type(genre_vec_list))
    return genre_vec_list
    


def getPlotVectors(df):
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

        # Trim strings before proceeding
        df['Title'] = df['Title'].str.strip()
        df['Genre'] = df['Genre'].str.strip()
        df['Plot'] = df['Plot'].str.strip()

        # Remove text in brackets
        df['Plot'] = df['Plot'].str.replace('\([\s\S]*\)', "")
        df['Plot'] = df['Plot'].str.replace('\[[\s\S]*\]', "")

        for i, item in df.iterrows():
            has_genre = False;
            has_plot = False;
            has_title = True;

            plot = item['Plot']
            title = item['Title']
            genre = item['Genre']


            ### TITLE ###

            ### METHOD 1 - remove titles ###
            ### loses data but removes foreign titles
            # Replace "duplicate" characters (weird hyphen, single 3-dots, non-breaking space, non-regular apostrophes, etc.)           
            title = title.replace('"', '')
            title = title.replace("–", "-")
            title = title.replace("…", "...")
            title = title.replace(" ", " ")
            title = title.replace("’", "'")
            title = title.replace("`", "'")
            title = title.replace("~", "")

            ### METHOD 2 - convert titles ###
            ### saves data but keeps foreign titles
            # Convert all titles to ASCII (no data loss)
            #title = unidecode(title)

            # Skip title if it contains non-ASCII characters -> avoid foreign titles
            if not(title.isascii()): continue


            ############################

            # Skip titles that do not contain any letters
            if not(bool(re.match('^(?=.*[a-zA-Z])', title))): continue

            # Skip non-english titles
            if not(is_in_english(title)): continue


            ### GENRE ###

            # Extract the predefined genres and write it with a consistent format into the new csv
            genre = genre.lower()
            new_genre = []
            for key in substrings:
                if key in genre:
                    new_genre.append(substrings[key])
                    has_genre = True;


            ### PLOT ###

            # Convert to ASCII to avoid non-ASCII characters
            plot = unidecode(plot)

            # Check if the plot is valid
            if (plot != ''):
                has_plot = True
                plot = plot.replace("\r\n", " ")
                plot = plot.replace("\n", " ")
                plot = plot.strip('\"')
                plot = plot.replace(";", ",")


                if "(" in plot or ")" in plot or "[" in plot or "]" in plot:
                    print("TITLE:   ", title)
                    print("PLOT:    ", plot)

            # Check if all three are valid
            if (has_genre and has_plot and has_title):
                title = str("\"" + title + "\"")
                new_genre = str("\"" + " ".join(new_genre) + "\"")
                plot = str("\"" + plot + "\"")

                csvfile.write(title + ";" + str(new_genre) + ";" + plot + "\n")

    df = pd.read_csv('modified_ds.csv', sep=';')
    print("Modified dataset shape: ", df.shape)

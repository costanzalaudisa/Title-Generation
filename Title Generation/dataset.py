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
    vocab = [k for k,c in vocab.items() if c >= 50]
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
    genre_list_10 = [
        "action", "science-fiction", "drama", "comedy",
        "horror", "thriller", "crime", "western", "adventure", "music"]
    genre_list_red = [
        "science-fiction", "drama", "horror", "crime", "western"]
    genre_vec_list = []

    # Set the genres we want to have as vector dimensions. 10 for the full 10 genres, red for the 5 best genres.
    genre_list = genre_list_red


    for i, item in df.iterrows():

    # Go through the substrings and check which genres apply fo the current row
        genre_vec = [0] * len(genre_list)
        genre = item['Genre']

        if (genre not in genre_list):
            print(genre)
            continue

        for index in range(len(genre_list)):
            if genre_list[index] in genre:
                genre_vec[index] = 1

        genre_vec_list.append(genre_vec)

    print(type(genre_vec_list))

    genre_vec_df = pd.DataFrame.from_records(genre_vec_list)
    return genre_vec_df
    


def getPlotVectors(df):
    # Load the vocab for the bag-of-words representation of the plot

    genre_list_10 = [
            "action", "science-fiction", "drama", "comedy",
            "horror", "thriller", "crime", "western", "adventure", "music"]
    genre_list_red = [
            "science-fiction", "drama", "horror", "crime", "western"]

    # Set the genres we have in the output vectors. 10 for all of the genres, red for the 5 best ones.
    genre_list = genre_list_red

    vocab = load_doc("vocab.txt")
    vocab = vocab.split()
    vocab = set(vocab)
    tokenizer = Tokenizer()
    table = str.maketrans('', '', punctuation)
    stop_words = set(stopwords.words('english'))
    token_list = list()
    tokenstr_list = list()

    for i, item in df.iterrows():

        genre = item['Genre']

        # If the genre of the movie is not part of the wanted genres, skip this entry.
        if genre not in genre_list:
            continue

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

    # CountVectorizer vectorizes the plots and each entry is the number of times a word appears in the plot
    CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                        stop_words='english')
    Count_data = CountVec.fit_transform(tokenstr_list)

    # TF-IDF vectorizer takes into account how often a word appears in the whole data set
    TFIDFvec = TfidfVectorizer(use_idf=True, 
                        smooth_idf=True,  
                        ngram_range=(1,1),           # to use bigrams ngram_range=(2,2)
                        stop_words='english')
    tf_idf_data = TFIDFvec.fit_transform(tokenstr_list)

    plot_frame   = pd.DataFrame(Count_data.toarray(),columns=CountVec.get_feature_names())
    tf_idf_frame = pd.DataFrame(tf_idf_data.toarray(),columns=TFIDFvec.get_feature_names())

    print(tf_idf_frame.iloc[:3].values)
    return tf_idf_frame

def writeCleanedCsv(df):

    # Define and sort genre list
    substrings_10 = {
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

    substrings_red = {
        "sci"       : "science-fiction",
        "dram"      : "drama",
        "hor"       : "horror",
        "crim"      : "crime",
        "west"      : "western",
        }

    genre_counter_10 = {
                    "action" : 0,
                    "science-fiction" : 0,
                    "drama" : 0,
                    "comedy" : 0,
                    "romance" : 0,
                    "horror" : 0,
                    "thriller" : 0,
                    "crime" : 0,
                    "western" : 0,
                    "fantasy" : 0,
                    "adventure": 0,
                    "music" : 0
                    }

    genre_counter_red = {
                    "science-fiction" : 0,
                    "drama" : 0,
                    "horror" : 0,
                    "crime" : 0,
                    "western" : 0,
                    }

    # Set the genre counter and substring. 10 is for the full 10 genres, red is for only 5 well performing genres.
    genre_counter = genre_counter_red
    substrings = substrings_red

    file_name = "modified_ds_red.csv"

    with open(file_name, 'w', encoding='utf-8') as csvfile:

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

            # Skip title if it contains non-ASCII characters -> avoid foreign titles
            #if not(title.isascii()): continue


            ############################

            # Skip titles that do not contain any letters
            # if not(bool(re.match('^(?=.*[a-zA-Z])', title))): continue

            # Skip non-english titles
            # if not(is_in_english(title)): continue


            ### GENRE ###

            # Extract the predefined genres and write it with a consistent format into the new csv
            genre = genre.lower()
            new_genre = []
            for key in substrings:
                if key in genre:
                    new_genre.append(substrings[key])
                    has_genre = True

            if (len(new_genre) > 1):
                has_genre = False
            

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

            # Check if all three are valid
            if (has_genre and has_plot and has_title):
                out_title = str("\"" + title + "\"")
                genre_counter[new_genre[0]] += 1
                out_genre = str("\"" + new_genre[0] + "\"")
                out_plot = str("\"" + plot + "\"")

                csvfile.write(out_title + ";" + str(out_genre) + ";" + out_plot + "\n")
        print(genre_counter)

    df = pd.read_csv(file_name, sep=';')
    print("Modified dataset shape: ", df.shape)

def balanceDataSet(filename):

    # Check how many times each genre is present in the data set, and depending on how big the differences is, remove some of them
    # Removing the genres "romance" and "fantasy, since these have very few entries
    # max_items is the threshold

    max_items = 900

    new_genre_counter_10 = {
                "action" : 0,
                "science-fiction" : 0,
                "drama" : 0,
                "comedy" : 0,
                "horror" : 0,
                "thriller" : 0,
                "crime" : 0,
                "western" : 0,
                "adventure": 0,
                "music" : 0
                }
    genre_counter_10 = {
                "action" : max_items,
                "science-fiction" : max_items,
                "drama" : max_items,
                "comedy" : max_items,
                "horror" : max_items,
                "thriller" : max_items,
                "crime" : max_items,
                "western" : max_items,
                "adventure": max_items,
                "music" : max_items
                }

    new_genre_counter_red = {
                "science-fiction" : 0,
                "drama" : 0,
                "horror" : 0,
                "crime" : 0,
                "western" : 0,
                }
    genre_counter_red = {
                "science-fiction" : max_items,
                "drama" : max_items,
                "horror" : max_items,
                "crime" : max_items,
                "western" : max_items,
                }

    # Set the genre counters. 10 is for the full 10 genres, red is for only 5 well performing genres.
    new_genre_counter = new_genre_counter_red
    genre_counter = genre_counter_red

    df = pd.read_csv(filename, sep=';')

    output_filename = "balanced_ds_red.csv"

    with open(output_filename, 'w', encoding='utf-8') as csvfile:
        csvfile.write("Title;Genre;Plot\n")

        for i, item in df.iterrows():
            title = item['Title']
            genre = item['Genre']
            plot = item['Plot']
            if genre in genre_counter and genre_counter[genre] > 0:
                csvfile.write(title + ";" + genre + ";" + plot + "\n")
                genre_counter[genre] -= 1
                new_genre_counter[genre] += 1

    print(new_genre_counter)
    df = pd.read_csv('balanced_ds.csv', sep=';')
    print("Balanced dataset shape: ", df.shape)

def combineDataFrames(X, Y):

    # X are the input variables, Y is the target

    new_df = pd.concat([X,Y], axis=1)
    return new_df
from imports import *
from dataset import prepare_dataset
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
#df = prepare_dataset()
df = pd.read_csv('wiki_movie_plots_deduped.csv')
print(df)

# Define and sort genre list
genres = df['Genre'].unique()
sorted_genres = sorted(genres)
print("Length of genre list:", len(sorted_genres), "  |   Genre list: ", sorted_genres)

action_list = []
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
    "musical"           : ["music"]
    }

for item in df.iterrows():

    for sub in substrings:


print(action_list)
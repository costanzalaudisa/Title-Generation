from imports import *
from dataset import *
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')

#df = prepare_dataset()
#vocab = build_vocab(df)

write_genre_csv(df)
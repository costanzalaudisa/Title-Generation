from imports import *
from dataset import *
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')
print("Original dataset shape:", df.shape)

#df = splitGenres(df) # not in use at the moment
vocab = buildVocab(df)

writeCleanedCsv(df)

#df = pd.read_csv('modified_ds.csv', sep=";")
#print("Modified dataset shape:", df.shape)

#getTitleVectors(df)

#getPlotVectors(df)

#getGenreVectors(df)
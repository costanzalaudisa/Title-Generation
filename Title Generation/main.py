from imports import *
from dataset import *
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
df = pd.read_csv('wiki_movie_plots_deduped.csv')

#df = prepareDataset()
#vocab = buildVocab(df)

#writeCleanedCsv(df)

df = pd.read_csv('modified_ds.csv', sep=";")

#getPlotVectors(df)

getGenreVectors(df)
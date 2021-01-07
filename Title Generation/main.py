from imports import *
from dataset import *
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
#df = prepare_dataset()
df = pd.read_csv('wiki_movie_plots_deduped.csv')

vocab = build_vocab(df)


print(vocab)
print(df)

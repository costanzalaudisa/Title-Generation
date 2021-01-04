from imports import *
from dataset import prepare_dataset
 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Gather dataset
df = prepare_dataset()
print(df)

# Define and sort genre list
genres = df['Genre'].unique()
sorted_genres = sorted(genres)
print("Length of genre list:", len(sorted_genres), "  |   Genre list: ", sorted_genres)
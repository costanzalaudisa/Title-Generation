from imports import *
from dataset import prepare_dataset

# 
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

df = prepare_dataset()
print(df)

genres = df['Genre'].unique()
print("Length of genre list:", len(genres), "  |   Genre list: ", genres)
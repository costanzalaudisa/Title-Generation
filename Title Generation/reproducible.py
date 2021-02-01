### Fix random seeds for reproductible results (see Keras FAQ) ###
seed_value = 123

# Avoid using GPU (comment if you want to use it)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# Set PYTHONHASHSEED at fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Set Python pseudo-random generator at fixed value
import random as rn
rn.seed(seed_value)

# Set Numpy pseudo-random generator at fixed value
import numpy as np
np.random.seed(seed_value)

# Set Tensorflow pseudo-random generator at fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
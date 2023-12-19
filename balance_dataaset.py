#%%
# balance the dataset of bollywood actors to have equal males and females
import os
from collections import Counter
import random
random.seed(42)

# read the dictionary about the sexes from a file
with open('sex_dictionary', 'r') as f:
    sex_dict = eval(f.read())

counts = Counter(sex_dict.values())

# since the number of 0 is greater than the number of 1 (99 vs 36),
# we will have to balance the dataset, we will do this by removing 
# 63 of the zeros chosen randomly.

num_zeros = counts[0]
num_ones = counts[1]
num_zeros_to_remove = num_zeros - num_ones

# remove num_zeros_to_remove zeros

males = [k for k in sex_dict if sex_dict[k] == 0]
keys_to_remove = random.sample(males, num_zeros_to_remove)

# remove the keys from the dictionary
for k in keys_to_remove:
    sex_dict.pop(k)
# print(Counter(sex_dict.values())) # it is balanced now

# move the appropriate folders in the sex_dict to a new folder called 'balanced'

if not os.path.exists('balanced'):
    # If not, create it
    os.makedirs('balanced')

# move the folders
for folder in os.listdir(os.path.join('Bollywood Actor Images')):
    if folder in keys_to_remove:
        continue
    else:
        os.rename(os.path.join('Bollywood Actor Images', folder),
                  os.path.join('balanced', folder))
print('done')
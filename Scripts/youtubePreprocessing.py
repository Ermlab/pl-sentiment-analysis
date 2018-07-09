"""

Simple script for preprocessing youtube comments data.
This example show you how to clean data from negative comments.

If you want clean positive comments replace:
columns = ['id','user','date','timestamp','likes']
data['rate'] = 1
output = save as other name, such as mergetYT2.csv

"""

# Import necessary packages

import pandas as pd
import numpy as np

# Read data
data = pd.read_csv('mergedYT.csv', delimiter=',')

# Select unused columns
columns = ['date','hasReplies','id','likes','numberOfReplies','replies.commentText','replies.date','replies.id','replies.likes',
           'replies.timestamp','replies.user','timestamp','user']


# Drop unused columns
data.drop(columns, inplace=True, axis=1)

# Add new column 'rate' as label for our neural network

data['rate'] = -1

# Get the length of text comments - add new column 'length' into CSV file
data['length'] = data['commentText'].str.len()

# Drop rows where len of strings are less than 20
data = data[data['commentText'].str.len() > 20]

# Drop URLs
data['commentText'] = data['commentText'].str.replace(r'http([^\s]+)','').astype('str')

# Drop words with #Hashtag
data['commentText'] = data['commentText'].str.replace(r'#([^\s]+)', '').astype('str')

# Drop Emoji from text and 'text images'
data['commentText'] = data['commentText'].str.replace(r'[^\w\s,]', '').astype('str')

# Remove lead white spaces
data['commentText'] = data['commentText'].str.lstrip()

# Rename commentText -> description for our neural network
data.rename(columns={'commentText':'description'}, inplace=True)

print(data)

# Save data after preprocessing
output = data.to_csv('mergedYT1.csv', encoding='utf8', index=False, header=True)
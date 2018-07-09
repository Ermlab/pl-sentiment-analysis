from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import models


model = load_model('finalsentimentmodel.h5')
model.summary()

with open('finalwordindex.pkl', 'rb') as picklefile:
    word_index = pickle.load(picklefile)
top_words = len(word_index)
tokenizer = Tokenizer(num_words=top_words)
tokenizer.word_index = word_index
print(word_index)


print('Found %s uniqe tokens.' % len(word_index))

# Insert text for example 'your sentence in Polish'
text = ['spierdalaj szmato']

test_sequences = tokenizer.texts_to_sequences(text)

x_test = sequence.pad_sequences(test_sequences, maxlen=40)

print('x_test shape:', str(x_test.shape))

result = model.predict(x_test)

print("Neutral: %.2f%%" % (result[:,0]*100))
print("Positive: %.2f%%" % (result[:,1]*100))
print("Negative: %.2f%%" % (result[:,2]*100))
#print(result)


model.load_weights('finalsentimentmodel.h5')

"""
X = tokenizer.texts_to_sequences(["Kaczyński byłby najlepszym premierem"])
X = sequence.pad_sequences(X)

pred = model.predict(X)
print(pred)
"""
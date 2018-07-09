# API for Polish sentiment analysis using Keras and Word2vec

Sentiment analysis is a natural language processing (NLP) problem where text is understood and the underlying intent is predicted.

I will show you how you can predict the sentiment of Polish language texts as either positive, neutral or negative in Python using the Keras Deep Learning library and Google Word2vec.

## Getting started



First of all you need to make sure you have installed Python 3.6. For that purpose we recommend Anaconda, it has all the necessary libraries except:
* scikit-learn 0.19.1
* Pandas 0.22.0
* NumPy 1.14.0
* Keras 2.1.4
* gensim 3.4.0
* many_stop_words 0.2.2
* TensorFlow 1.6.0
* wordcloud 1.4

All libraries can be installed with the following commands:

```
pip install scikit-learn
pip install Keras
pip install gensim
pip install many_stop_words
pip install TensorFlow
pip install wordcloud
```

or quickly:
```
pip install -r requirements.txt
```


Once you have installed Python and the dependencies download at least pre-trained Polish Word Embedding model
[here](http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-cbow-hs.txt.gz) and extract to main project directory.

The easiest way to see our method in action is to run the LSTM.py script.

## Data

Download our dataset from [Google Drive](https://drive.google.com/open?id=1P87kDKspU8n6V7iHl1Pd-qgT0c-n3VlE) and extract to /Data directory.

Our dataset was collected from various sources:

1. [Opineo](opineo.pl) - Polish service with all reviews from online shops
2. Twitter - Polish current top hashtags from political news and Polish Election Campaign 2015
3. [Polish Academy of Science HateSpeech project](http://zil.ipipan.waw.pl/HateSpeech)
4. YouTube - comments from various videos

Download Polish Word Embeddings from Polish Academy of Science from [Google Drive](https://drive.google.com/open?id=1LLB0p61b2dk-JJt82n96yhVTSj2ORQsI)
and extract it in main folder.

## Models

Our pre-trained models you can download from [Google Drive](https://drive.google.com/file/d/1avgoKihXVpe16rAGGV1x0j11h5Mjij-e/view?usp=sharing):
* finalsentimentmodel.h5
* finalwordindex.pkl

and extract it in Models/


## Contact

* Main author: [Szymon Płotka](https://github.com/simongeek)
* CEO of Ermlab Software [Krzysztof Sopyła](https://github.com/ksopyla)
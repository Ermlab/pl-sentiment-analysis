# API for Polish sentiment analysis using Keras and Word2vec

Sentiment analysis is a natural language processing (NLP) problem where text is understood and the underlying intent
is predicted.

I will show you how you can predict the sentiment of Polish language texts as either positive, neutral or negative
in Python using the Keras Deep Learning library and Google Word2vec.

Check Our blog post [Polish sentiment analysis using Keras and Word2vec](https://ermlab.com/en/blog/nlp/polish-sentiment-analysis-using-keras-and-word2vec/)

## Getting started



First of all you need to make sure you have installed Python 3.6. For that purpose we recommend Anaconda,
it has all the necessary libraries except:
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
[here](http://dsmodels.nlp.ipipan.waw.pl/dsmodels/nkjp+wiki-forms-all-100-cbow-hs.txt.gz) and extract
to main project directory.

The easiest way to see our method in action is to run the LSTM.py script.

## Data

Download our dataset from [Google Drive](https://drive.google.com/file/d/1vXqUEBjUHGGy3vV2dA7LlvBjjZlQnl0D/view?usp=sharing)
and extract to /Data directory.

Our dataset was collected from various sources:

1. [Opineo](opineo.pl) - Polish service with all reviews from online shops
2. Twitter - Polish current top hashtags from political news and Polish Election Campaign 2015
3. [Polish Academy of Science HateSpeech project](http://zil.ipipan.waw.pl/HateSpeech)
4. YouTube - comments from various videos

Download [Polish Word Embeddings from Polish Academy of Science](http://dsmodels.nlp.ipipan.waw.pl/w2v.html)
and extract it in main folder.


## Useful repos

* https://github.com/Kyubyong/wordvectors, pre-trained word vector models for non-English languages
* https://github.com/dakshitagrawal97/TweetSentimentAnalysis - jest zgoda!!
* https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
* http://dsmodels.nlp.ipipan.waw.pl/
* https://github.com/BUPTLdy/Sentiment-Analysis
* https://github.com/shayanzare007/EntitySentiment
* https://github.com/Theo-/sentiment-analysis-keras-conv
* https://github.com/giuseppebonaccorso/twitter_sentiment_analysis_word2vec_convnet
* https://github.com/PAS43/TwitterSentimentAnalysis
* https://github.com/kailashnathan
* https://github.com/ankeshanand/review-helpfulness
* https://github.com/spopov812/Sentiment-Analysis-Deep-Recurrent-Neural-Network-LSTM-
* https://github.com/xingziye/movie-reviews-sentiment
* https://github.com/dswald/sentiment_analysis/tree/0d6a85e4afef33eef5da7e6ac8aa2dbdb5e89de2
* https://github.com/sukilau/amazon-sentiment-analysis


## Contact & blog post

* Main author: [Szymon Płotka](https://github.com/simongeek)
* CEO of [Ermlab Software](https://ermlab.com) [Krzysztof Sopyła](https://github.com/ksopyla)
* check Our blog post [Polish sentiment analysis using Keras and Word2vec](https://ermlab.com/en/blog/nlp/polish-sentiment-analysis-using-keras-and-word2vec/)

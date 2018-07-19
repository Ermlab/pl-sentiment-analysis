#!/bin/bash

echo "Grab word2vec data, some files are really big >1GB"


wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_50.model
wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_50.model.syn0.npy
wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_50.model.syn1neg.npy
wget http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_50.pkl

#wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_100.model
#wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_100.model.syn0.npy
#wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_100.model.syn1neg.npy
#wget http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_100.pkl

# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_200.model
# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_200.model.syn0.npy
# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_200.model.syn1neg.npy
# #wget http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_200.pkl

# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_300.model
# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_300.model.syn0.npy
# wget -N http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_300.model.syn1neg.npy
# #wget http://mozart.ipipan.waw.pl/~axw/models/lemma/w2v_allwiki_nkjpfull_300.pkl

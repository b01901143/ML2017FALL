#!/bin/bash
wget 'https://www.dropbox.com/s/hb3fpwb8rv989rj/embedding?dl=0' -O 'embedding'
wget 'https://www.dropbox.com/s/zagcquxhrt00fqf/embedding.syn1neg.npy?dl=0' -O 'embedding.syn1neg.npy'
wget 'https://www.dropbox.com/s/dp43h7t3tvzq31p/embedding.wv.syn0.npy?dl=0' -O 'embedding.wv.syn0.npy'
python3 punc_train.py $1


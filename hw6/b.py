import jieba
from keras.preprocessing.text import text_to_word_sequence
import matplotlib.pyplot as plt
import matplotlib as mpl
import gensim
from sklearn.manifold import TSNE
from adjustText import adjust_text

font_name = "STKaiti"
mpl.rcParams['font.family']=font_name

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for i,word in enumerate(model.wv.vocab):
        if i>100:
            break
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, verbose=1)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.savefig('wordvec.png')

jieba.set_dictionary('dict.txt.big')
'''
context = []
texts = open('data/all_sents.txt')
for line in texts:
    seg_list = jieba.cut(line, cut_all=False)
    context.append(" ".join(seg_list))
    context[-1] = text_to_word_sequence(context[-1], filters='\n')

model = gensim.models.Word2Vec(context, min_count=1, size=400, sg=0, workers=32, iter=10)
model.save('embedding')
'''
model = gensim.models.Word2Vec.load('embedding')
tsne_plot(model)


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec_Ngram
import codecs


class MySentences(object):
    def __init__(self):
        pass
 
    def __iter__(self):
        wiki_data = codecs.open('English/Wikipedia_data/wiki_en.txt', encoding='utf-8')
        for row in wiki_data:
            yield(row.lower().split())

sentences = MySentences()
w2v_params = {
    'alpha': 0.025,
    'size': 128,
    'window': 7,
    'iter': 1,
    'min_count': (5,10,10),
    'sample': 1e-4,
    'sg': 1,
    'hs': 0,
    'negative': 10,
    'word_gram' : (1,2),
    'context_gram' : (1,2),
    'workers' : 10

}

model = Word2Vec_Ngram(sentences, **w2v_params) 
model.save("model_wiki_amazon_bigram")



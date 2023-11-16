import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem_word(word):
    return stemmer.stem(word).lower()
    

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem_word(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32)
    
    for word in tokenized_sentence:
        if word in all_words:
            bag[all_words.index(word)] = 1.0
            
    return bag
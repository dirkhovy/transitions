import codecs
import sys
import nltk.data
from nltk.tokenize import word_tokenize

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for line in codecs.open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    sent_tokenize_list = tokenizer.tokenize(line)
    for sent in sent_tokenize_list:
        words = word_tokenize(sent)
        if len(words) > 1:
            print(codecs.encode(" ".join(words), "utf-8"))
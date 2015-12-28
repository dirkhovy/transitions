import codecs
import argparse

parser = argparse.ArgumentParser(description="filter out all pairs of spelling variants where both are in the dictionary (requires list of misspelled words)")
parser.add_argument('input', help='input text file')
parser.add_argument('--misspelled', help='list of misspelled words')

args = parser.parse_args()

misspelled = set(map(str.strip, codecs.open(args.misspelled, encoding='utf-8').readlines()))

for line in codecs.open(args.input, encoding='utf-8'):
    line = line.strip()
    word1, word2, w = line.split('\t')
    if word1 in misspelled or word2 in misspelled:
        if not word1 in word2 and not word2 in word1:
            print(codecs.encode(line, 'utf-8'))

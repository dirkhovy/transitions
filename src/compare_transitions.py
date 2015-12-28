import nltk

__author__ = 'dirkhovy'

import argparse
import codecs
from collections import defaultdict, Counter
import glob
import numpy as np
import sys
import scipy.stats as sps
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="get differences between n-gram distros for two or more annotated systems")
parser.add_argument('--input', help='sets of files (can be wildcards), comma-separated', required=True)
parser.add_argument('--names', help='systems names, comma-separated', required=False)
parser.add_argument('--n', help='size of n-grams', type=int, default=2)

args = parser.parse_args()

names = args.names.split(',') if args.names else list(range(len(args.input.split(','))))

def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            yield (current_words, current_tags)
            current_words = []
            current_tags = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        yield (current_words, current_tags)


def get_distro(files, n):
    counts = Counter()
    for file_name in sorted(glob.glob(files)):
        print(file_name, file=sys.stderr)
        for words, tags in read_conll_file(file_name):
            ngrams = nltk.ngrams(tags, n)
            counts.update(ngrams)
    return counts


counts = []
for files in args.input.split(','):
    system_counts = get_distro(files, args.n)
    counts.append(system_counts)

all_keys = Counter()
for system_counts in counts:
    all_keys.update(system_counts)


# compare all systems with each other
for i, sys1 in enumerate(counts):
    s1_total = float(sum(sys1.values()))
    s1 = np.array([sys1[key]/s1_total for key in sorted(all_keys.keys())])
    s1 += 1

    for j, sys2 in enumerate(counts):
        if i != j:
            fig, ax = plt.subplots()

            s2_total = float(sum(sys2.values()))
            s2 = np.array([sys2[key]/s2_total for key in sorted(all_keys.keys())])
            s2 += 1

            width = 0.5
            # use KL
            print(names[i], names[j], sps.entropy(s1, s2))

            # plot distros
            N = np.array(list(range(len(s1))))
            bar1 = ax.bar(N, s1, width=width/2, color='r')
            bar2 = ax.bar(N+width/2, s2, width=width/2, color='b')

            ax.set_xticks(N+width)
            ax.set_xticklabels(sorted(all_keys.keys()))
            ax.legend( (bar1[0], bar2[0]), (names[i], names[j]) )

            # differences = zip(np.abs(s1-s2), sorted(all_keys.keys()))
            differences = sorted(zip(s1-s2, sorted(all_keys.keys())))
            print(names[j], ', '.join(('--'.join(key) for f, key in differences[:5])))
            print(names[i], ', '.join(('--'.join(key) for f, key in differences[-5:])))
            print()

            # plt.show()


            # plot hinton diagrams

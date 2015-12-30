import argparse
import sys
import nltk
import re
from collections import defaultdict, Counter

numbers = re.compile(r"[0123456789]")

parser = argparse.ArgumentParser(
    description="create FSAs from a CoNLL style file, one word with possible tags per line")
parser.add_argument('--fsa', help='make FSA', action="store_true")
parser.add_argument('--fst', help="make FST", action="store_true")
parser.add_argument('--data', help="print out data", action="store_true")
parser.add_argument('--trigrams', help="use trigram transitions", action="store_true")
parser.add_argument('--train', help='training data in CoNLL data plus info')
parser.add_argument('--test', help='training data in CoNLL data plus info')
parser.add_argument('--conll', help='CoNLL data')
parser.add_argument('--dictionary', help='wiktionary with allowed tags for words')
parser.add_argument('--prefix', help="output prefix (default: automaton)", default='automaton')

args = parser.parse_args()

"""
- read in test, record the emission parameters
- read in dict, record the emission parameters
- read in CoNLL, record the transition parameters
- read in train
"""

# TODO: update to UPOS2
TAGS = set(['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X'])
AGE_GROUPS = set(['U35', 'O45'])
SMOOTHING = 0.000000000001


def read(fname, target):
    """
    read in a CoNLL style file
    format of files: each line is
    "word<TAB>tag<newline>", followed by
    "age,gender",
    blank line is new sentence.
    :param fname: file to read
    :return: generator of ([words, tags], [labels])
    """
    sentence = []
    tags = []
    labels = None
    include = True

    for line in open(fname):
        line = line.strip()

        if not line:
            if sentence != [] and include:
                yield (sentence, tags, labels)
            sentence = []
            tags = []
            labels = None
            include = True

        else:
            elements = line.split('\t')

            # read in age and gender info
            if len(elements) == 1:
                age, gender = line.split(',')

                if target == 'age':
                    if age == 'NONE':
                        include = False
                    labels = age
                elif target == 'gender':
                    if gender == 'NONE':
                        include = False
                    labels = gender
                elif target == 'joint':
                    labels = '%s-%s' % (age, gender)
                    if age == 'NONE' or gender == 'NONE':
                        include = False
                elif target == 'both':
                    labels = [age if age != 'NONE' else None, gender if gender != 'NONE' else None]

            # read in words and tags
            elif len(elements) == 2:
                word, pos_tag = elements
                word = word.replace('"', "''")
                word = word.replace('\\', '')
                word = re.sub(numbers, '0', word)
                sentence.append(word)
                tags.append(pos_tag)

            else:
                print('Problem reading input file "%s": unexpected line "%s"' % (fname, line))


begin = defaultdict(int)
end = defaultdict(int)
emissions = defaultdict(lambda: defaultdict(int))
transitions = defaultdict(lambda: defaultdict(int))
trigrams = defaultdict(lambda: defaultdict(int))
end_trigrams = defaultdict(int)
word_count = Counter()
in_dict = set()

if args.data:
    test_file = open('%s.test' % args.prefix, 'w')
    test_key_file = open('%s.test.key' % args.prefix, 'w')
    test_age_file = open('%s.test.age' % args.prefix, 'w')

    dev_file = open('%s.dev' % args.prefix, 'w')
    dev_key_file = open('%s.dev.key' % args.prefix, 'w')
    dev_age_file = open('%s.dev.age' % args.prefix, 'w')

    eval_file = open('%s.eval' % args.prefix, 'w')
    eval_key_file = open('%s.eval.key' % args.prefix, 'w')
    eval_age_file = open('%s.eval.age' % args.prefix, 'w')

    instances = defaultdict(list)

# read in test, record the emission parameters
# TODO: add in UNKs here...
print("reading test", file=sys.stderr)
for (sentence, tags, labels) in read(args.test, 'age'):
    word_count.update(sentence)

    if args.data:
        instances[labels].append((sentence, tags))

        test_file.write('\n"%s"\n' % ('" "'.join(sentence)))
        test_key_file.write('%s\n' % (' '.join(tags)))
        test_age_file.write('%s\n' % (labels))

    for (word, tag) in zip(sentence, tags):
        emissions[tag][word] += SMOOTHING
        in_dict.add(word)

if args.data:
    for label, items in instances.items():
        half = len(items)
        for i, (sentence, tags) in enumerate(items):
            if i < half:
                dev_file.write('\n"%s"\n' % ('" "'.join(sentence)))
                dev_key_file.write('%s\n' % (' '.join(tags)))
                dev_age_file.write('%s\n' % (label))
            else:
                eval_file.write('\n"%s"\n' % ('" "'.join(sentence)))
                eval_key_file.write('%s\n' % (' '.join(tags)))
                eval_age_file.write('%s\n' % (label))

    test_file.close()
    test_key_file.close()
    test_age_file.close()

    dev_file.close()
    dev_key_file.close()
    dev_age_file.close()

    eval_file.close()
    eval_key_file.close()
    eval_age_file.close()


# read in dict, record the emission parameters
print("reading dictionary", file=sys.stderr)
for line in open(args.dictionary):
    _, word, tag, _ = line.strip().split('\t')
    word = word.replace('"', "''")
    emissions[tag][word] += 1
    in_dict.add(word)

# read in CoNLL, record the transition parameters
print("reading CoNLL", file=sys.stderr)
for (sentence, tags, labels) in read(args.conll, 'age'):
    begin[tags[0]] += 1
    end[tags[-1]] += 1

    for (word, tag) in zip(sentence, tags):
        emissions[tag][word] += 1
        in_dict.add(word)

    for tag1, tag2 in nltk.bigrams(tags):
        transitions[tag1][tag2] += 1

    if args.trigrams:
        if len(tags) > 1:
            end_trigrams[(tags[-1], tags[-2])] += 1
        for (tag1, tag2, tag3) in nltk.trigrams(tags):
            trigrams[(tag1, tag2)][tag3] += 1

# read in train
print("reading train", file=sys.stderr)
for (sentence, tags, labels) in read(args.train, 'age'):
    word_count.update(sentence)

    for word in sentence:
        if word not in in_dict:
            for tag in TAGS:
                emissions[tag][word] += 1

hapaxes = set([word for word, count in word_count.items() if count == 1])
print(len(hapaxes), "hapaxes", file=sys.stderr)

# add smoothing to prevent empty parameters
for tag in TAGS:
    emissions[tag]['UNK'] += 1
    begin[tag] += 1
    end[tag] += 1
    for tag2 in TAGS:
        transitions[tag][tag2] += SMOOTHING

        if args.trigrams:
            end_trigrams[(tag, tag2)] += SMOOTHING
            for tag3 in TAGS:
                trigrams[(tag, tag2)][tag3] += SMOOTHING

if args.data:
    print("\nwriting train", file=sys.stderr)
    train_file = open('%s.train' % args.prefix, 'w')
    for (sentence, tags, labels) in read(args.train, 'age'):
        sentence = ['UNK' if word in hapaxes else word for word in sentence]
        train_file.write('"%s"\n"%s"\n' % (labels, '" "'.join(sentence)))
    train_file.close()

if args.fsa:
    print("writing FSA", file=sys.stderr)

    fsa = open('%s.fsa' % args.prefix, 'w')

    fsa.write('S\n')
    for tag, words in emissions.items():
        tag_total = sum(words.values())
        for word, count in words.items():
            word = word.replace('"', "''")
            for age_group in AGE_GROUPS:
                fsa.write('(S (S "%s_%s" "%s" %s))\n' % (tag, age_group, word, count / tag_total))
    fsa.close()

if args.fst:
    print("writing FST", file=sys.stderr)

    fst = open('%s.fst' % args.prefix, 'w')

    fst.write('END\n')
    # TODO: experiment with age-specific emissions - should not make a difference (seems to be in transitions)...
    for age_group in AGE_GROUPS:
        fst.write('(S (S-%s "%s" *e* 0.5))\n' % (age_group, age_group))
        # fst.write('(S (S_DECISION "%s" *e* 0.5))\n' % (age_group))
        # fst.write('(S_DECISION (S-%s *e* *e* 1))\n' % (age_group))

        begin_total = sum(begin.values())
        for tag, tag_count in begin.items():
            fst.write('(S-%s (%s-%s *e* "%s_%s" %s))\n' % (
            age_group, age_group, tag, tag, age_group, tag_count / begin_total))

        # bigram transitions
        if not args.trigrams:
            for tag1, tags in transitions.items():
                tag_total = sum(tags.values())
                for tag2, count in tags.items():
                    fst.write('(%s-%s (%s-%s *e* "%s_%s" %s))\n' % (
                    age_group, tag1, age_group, tag2, tag2, age_group, count / tag_total))

        # transitions and end for trigrams
        if args.trigrams:
            end_total = sum(end_trigrams.values())
            for (tag1, tag2), tags in trigrams.items():
                fst.write(
                    '(%s-%s (%s-%s-%s *e* "%s_%s" %s))\n' % (age_group, tag1, age_group, tag1, tag2, tag2, age_group,
                                                             transitions[tag1][tag2] / sum(
                                                                 transitions[tag1].values())))
                tag_total = sum(tags.values())
                for tag3, count in tags.items():
                    fst.write(
                        '(%s-%s-%s (%s-%s-%s *e* "%s_%s" %s))\n' % (
                        age_group, tag1, tag2, age_group, tag2, tag3, tag3, age_group, count / tag_total))

                fst.write(
                    '(%s-%s-%s (END *e* *e* %s))\n' % (age_group, tag1, tag2, end_trigrams[(tag1, tag2)] / end_total))

        # end for bigrams
        else:
            end_total = sum(end.values())
            for tag, count in end.items():
                fst.write('(%s-%s (END *e* *e* %s))\n' % (age_group, tag, count / end_total))

    fst.close()

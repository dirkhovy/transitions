import codecs
import sys

LIMIT = 0.3

current = []
unks = 0.0
for line in codecs.open(sys.argv[1], encoding='utf-8'):
    line = line.strip()

    if line:

        if line.startswith('UNK'):
            unks += 1.0
            current.append('UNK')
        else:
            word, tags = line.split('\t')
            if word == '"':
                word = "''"

            current.append(word)


    else:
        if unks / len(current) < LIMIT:
            print(codecs.encode('\n"%s"' % ('" "'.join(current)), 'utf-8'))
        unks = 0.0
        current = []
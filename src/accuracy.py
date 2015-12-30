import argparse
import re

numbers = re.compile(r"_[OU][34]5")


parser = argparse.ArgumentParser(
    description="get acuracy form prediction file")
parser.add_argument('--prediction', help='prediction file, no quotes, no empty lines')
parser.add_argument('--gold', help='gold file, no quotes, no empty lines')
args = parser.parse_args()

predictions = list(map(str.split, open(args.prediction)))
gold = list(map(str.split, open(args.gold)))

correct_sentences = 0
total_sentences = 0

correct_tokens = 0
total_tokens = 0

for predicted_sequence, gold_sequence in zip(predictions, gold):
    total_sentences += 1
    total_tokens += len(gold_sequence)

    # remove age suffix if existent
    predicted_sequence = [re.sub(numbers, '', token) for token in predicted_sequence]

    correct_sentences += int(predicted_sequence == gold_sequence)
    correct_tokens += sum([1 if predicted_word == gold_word else 0 for predicted_word, gold_word in
                           zip(predicted_sequence, gold_sequence)])

print("%.4f sentence accuracy; %.4f token accuracy" % (
correct_sentences / total_sentences, correct_tokens / total_tokens))

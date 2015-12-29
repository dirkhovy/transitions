#!/bin/bash
# USAGE: DATA AUTOMATA-BASE
DATA=$1
cp $2.fst $$FST
cp $2.fsa $$FSA

for i in $(seq 1 1 50)
do
    sh train_EM.sh 1 $DATA $$FST $$FSA
    cp $$FST.trained $$FST
    cp $$FSA.trained $$FSA

    mv $$FST.trained $2.iteration$i.fst
    mv $$FSA.trained $2.iteration$i.fsa

    sh predict.sh $2.iteration$i.fsa $2.iteration$i.fst data/input/english.test $2.iteration$i.test
    sh predict.sh $2.iteration$i.fsa $2.iteration$i.fst data/input/english.eval $2.iteration$i.eval
    sh predict.sh $2.iteration$i.fsa $2.iteration$i.fst data/input/english.dev $2.iteration$i.dev
done

rm $$FSA
rm $$FST

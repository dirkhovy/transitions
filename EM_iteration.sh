# USAGE: DATA AUTOMATA-BASE
DATA=$1
cp $2.fst $$FST
cp $2.fsa $$FSA

for i in {1..20}
do
    sh train_EM.sh 1 $DATA $$FST $$FSA
    cp $$FST.trained $$FST
    cp $$FSA.trained $$FSA

    mv $$FST.trained $2.iteration$i.fst
    mv $$FSA.trained $2.iteration$i.fsa
done

rm $$FSA
rm $$FST
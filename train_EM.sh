# USAGE: iterations DATA transition-FST emission-FSA
CARMEL="/Users/dirkhovy/Tools/graehl/carmel/bin/macosx/carmel"

$CARMEL -M $1 -f 0.000001 -aHJmZX 0.99999 --disk-cache-derivations --train-cascade $2 $3 $4

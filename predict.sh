#!/bin/bash
# FSA FST data
#CARMEL="/Users/dirkhovy/Tools/graehl/carmel/bin/macosx/carmel"
CARMEL="/home/dirkh/graehl/carmel/bin/linux64/carmel"

# predict tags
$CARMEL --project-right --project-identity-fsa -HJm $2 > $2.noe
grep -v "^$" $3| $CARMEL -qQbsriWIEk 1 $2.noe $1 > $4.prediction

# predict age
grep -v "^$" $3| $CARMEL -qQbsriWIEk 1 $2 $1 > $4.age.prediction

#cat $3| $CARMEL -qQbsriWIEk 1 $1 $2.noe

# Oppgave 4

## a - Many to Many LSTM
Ta utgangspunkt i rnn/generate-characters og tren modellen på
bokstavene “ hello world “. 

Bruk deretter modellen til å
generere 50 bokstaver etter inputen “ h”.

## b - Many to One LSTM
Tren modellen ulike ord (bruk fortsatt bokstavkoding som i
oppgave a) for emojis, for eksempel 
“hat ”: , “rat “: , “cat “: , “flat”: , “matt”: , “cap “: , “son ”: .

For å kunne trene i batches er ordene padded med mellomrom på slutten (hvis ordene er mindre enn makslengden).

Test deretter modellen på ord som “rt ” og “rats”, og se hvilken
emoji du får
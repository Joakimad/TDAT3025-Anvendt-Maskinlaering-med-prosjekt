#Oppgave 1

Ta gjerne utgangspunkt i [linear-2d](https://gitlab.com/ntnu-tdat3025/regression/linear-2d).

Datasettene i deloppgavene inneholder observasjoner om nyfødte barn. Det
kan være til hjelp å visualisere observasjonene først.
For alle deloppgavene: du skal visualisere modellen etter optimalisering
sammen med observasjonene, og skrive ut tapsverdien (loss) for modellen.

##a 
Lineær regresjon i 2 dimensjoner:
<br>Lag en lineær modell som predikerer vekt ut fra lengde gitt
observasjonene i length_weight.csv

##b
Lineær regresjon i 3 dimensjoner:
<br>Lag en lineær modell som predikerer alder (i dager) ut fra lengde og
vekt gitt observasjonene i day_length_weight.csv

##c
Ikke-lineær regresjon i 2 dimensjoner:
<br> Lag en ikke-lineær modell som predikerer hodeomkrets ut fra alder (i
dager) gitt observasjonene i day_head_circumference.csv
Bruk følgende modell prediktor: <br>f(x) = 20σ(xW + b) + 31
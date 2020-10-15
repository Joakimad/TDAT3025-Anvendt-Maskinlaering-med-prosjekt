# Oppgave 8

## 1. Q-Læring: Cartpole
Implementer Q-læring og bruk det for å løse cartpole-environmentet.

https://gym.openai.com/envs/CartPole-v0/

https://github.com/openai/gym

## 2. Q-Læring: Gridworld med visualisering
Lag et enkelt gridworld-environment. Dette innebærer at environmentet har et
diskret rutenett, og at en agent kan bevege seg rundt med fire handlinger (opp,
ned, høyre, venstre). 

Simuleringen terminerer når agenten har nådd et plassert
mål-posisjon som gir reward 1. Om man ønsker, kan det legges inn f.eks. solide
vegger eller farlige omr˚ader som gir straff rundt omkring. Environmentet skal
ha samme interface som cartpole (.step(a)-funksjon, og .reset())


Deretter skal implementasjonen av Q-læring fra forrige oppgave brukes for å
trene en agent i environmentet. Til slutt skal Q-verdiene visualiserer inne i selve
environmentet, og dette kan gjøres på flere måter. En måte er å fargelegge rutene
basert på den høyeste Q-verdien fra tilsvarende rad i Q-tabellen. Alternativt så
kan man tegne inn piler som peker i samme retning som handlingen med høyest
Q-verdi.

Tips: Biblioteket pygame er veldig greit for å lage visualisering av environmentet.
# Notes

## Log
- [20/11/2026-11:14] Setup completato. Obiettivo: Creare il progetto in locale, collegare Git, creare una repo su Github e collegare Colab. Il flusso di lavoro sarà
1. Scrivi e testa il codice in locale.
2. git push.
3. In Colab: !git pull e lancia il training.

## Cosa non ho capito
-Al momento tutto ok. Ho capito meglio come funzionano gli ambienti virtuali.

## Bug incontrati e come li ho risolti
-Al momento nessun bug.


# Notes

## Blocco 1A: PyTorch con sin(x)

### Setup
- Dati: 2000 punti casuali in [0, 2π], target sin(x), split 1600/400, seed fisso
- Modello: MLP 1→64→64→1, loss MSE, 200 epoche
- Ottimizzatore finale: Adam (lr tra 1e-3 e 5e-3)

### Bug incontrati e lezioni
1. **train_loop indentato dentro `if t%20==0`**: il modello faceva 10 epoche
   reali su 200. Nessun errore, risultati plausibili ma sbagliati.
   Lezione: contare sempre gli step di ottimizzazione effettivi.
2. **SGD con lr=1e-3**: convergenza troppo lenta. Passato ad Adam, molto più
   robusto al learning rate.
3. **Residuo del tutorial di classificazione** (`correct`, `argmax`) nel
   test loop: privo di senso in regressione e crea un broadcasting (N,N)
   silenzioso. Rimosso.
4. **Conseguenza**: il primo confronto ReLU/tanh era fatto con il loop
   sbagliato, quindi non valido. Da rifare.
   Lezione: dopo aver trovato un bug, rivalutare i risultati precedenti.

### Risultati
- **Gradiente tramite autograd in x=1**: 0.5376 vs cos(1)=0.5403
  (errore ~0.5%). La rete ha imparato anche la derivata senza vederla.
- **Riproducibilità**: stesso seed → stessa loss a tutte le cifre.
  Per misurare la variabilità servono più seed (modello e ottimizzatore
  ricreati a ogni run).
- **CPU vs GPU** (5 ripetizioni, 200 epoche, batch 64, 5000 step):
  - CPU: 8.8–9.7 s
  - GPU (T4): 10.7–11.5 s
  - La CPU è ~15-20% più veloce. Non è rumore: gli intervalli non si
    sovrappongono.
  - Causa: con una rete così piccola domina l'overhead (Python, lancio dei
    kernel, DataLoader che indicizza campione per campione).
  - TODO: batching manuale senza DataLoader; batch 1024 con rete larga
    (512 unità) per vedere la GPU vincere.

- **Overfitting su 10 punti** (tanh, Adam lr=5e-3, 2000 epoche):
  train 4.7e-1 → 9e-10, test ~2.8e-3. Il modello memorizza (codice sano)
  ma non generalizza perfettamente. Test leggermente peggiore dopo
  l'epoca ~1000. Picchi di train (epoche 601, 1601) dovuti a lr alto.
  Il gap è moderato perché sin è liscia: con soluzioni con shock sarà
  molto più marcato.
- **Bug del test di overfitting**: ottimizzatore creato su `model_local`
  ma ciclo eseguito su `model` → nessun aggiornamento, loss identica a
  tutte le cifre. Segnale di allarme: loss uguale tra stampe diverse.

### Cosa ho imparato
- La GPU conviene solo quando il lavoro per kernel è abbastanza grande da
  coprire l'overhead di lancio.
- Con dati piccoli conviene tenere tutto il dataset sul device.
- Un singolo run non dice nulla sulla variabilità: servono più seed.
- Debug: overfittare pochi campioni è il primo test per un modello nuovo.
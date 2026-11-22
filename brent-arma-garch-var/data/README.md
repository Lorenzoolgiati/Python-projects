# Dati

I prezzi del Brent non sono inclusi nel repository: provengono da Yahoo Finance e si aggiornano ogni giorno.

Per scaricarli ed eseguire l'analisi:

```bash
python main.py
```

Lo script salva i prezzi di chiusura in `data/brent.csv`. Le esecuzioni successive possono riutilizzarli senza connessione:

```bash
python main.py --offline
```

- **Ticker:** `BZ=F` (future continuo sul Brent, contratto front-month)
- **Periodo:** dal 1 gennaio 2010 a oggi
- **Frequenza:** giornaliera

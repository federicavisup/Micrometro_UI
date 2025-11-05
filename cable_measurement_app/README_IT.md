# Sistema di Studio Fattibilità - Misura Diametro Cavi Metallici

Applicazione Python con interfaccia PyQt6 per la misurazione ortogonale del diametro di cavi metallici con stima ML del diametro equivalente.

## Caratteristiche

- **Acquisizione dati in tempo reale** (10 Hz) di misure ortogonali Dx e Dy
- **Grafico dinamico** che mostra l'andamento delle misure nel tempo
- **Stima ML del diametro equivalente** tramite media geometrica con correzione
- **Calcolo automatico del peso per metro** del cavo in acciaio (ρ = 7850 kg/m³)
- **Metriche di confronto** complete (Bias, RMSE, MAE, Accuratezza, Errore %)
- **Controlli intuitivi** per avvio/stop acquisizione e regolazione finestra temporale

## Requisiti

- Python 3.8 o superiore
- PyQt6
- pyqtgraph
- numpy

## Installazione

1. Assicurati di avere Python installato sul tuo sistema

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

Oppure installa manualmente:
```bash
pip install PyQt6 pyqtgraph numpy
```

## Utilizzo

Avvia l'applicazione con:
```bash
python cable_measurement_app.py
```

### Funzionalità principali

1. **Avviare l'acquisizione**: Clicca su "▶ Avvia Acquisizione" per iniziare a raccogliere dati
2. **Fermare l'acquisizione**: Clicca su "⏹ Arresta Acquisizione" per terminare
3. **Regolare la finestra temporale**: Usa lo slider per modificare l'intervallo di visualizzazione (1-10 secondi)
4. **Confrontare con valore atteso**: 
   - Inserisci il valore atteso in kg/m
   - Clicca su "Calcola Metriche"
   - Visualizza le metriche di confronto (Bias, Errore %, RMSE, MAE, Accuratezza)

### Metriche calcolate

- **Dx e Dy**: Misure ortogonali del diametro in tempo reale
- **Diametro Equivalente (ML)**: Stima ottenuta tramite modello ML
- **Peso per Metro**: Calcolato con la formula π × (D/2)² × ρ
- **Bias**: Errore sistematico rispetto al valore atteso
- **Errore %**: Errore percentuale
- **Errore Assoluto**: Valore assoluto della differenza
- **Accuratezza**: Precisione della misura in percentuale
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

## Simulazione dei dati

L'applicazione simula la misurazione di un cavo con diametro base di ~10mm con una variazione realistica del 5%. I dati vengono acquisiti a 10 Hz (ogni 100ms).

## Formula per il calcolo del peso

```
Peso/metro = π × (D/2)² × ρ
```

Dove:
- D = diametro in mm
- ρ = densità dell'acciaio = 7850 kg/m³

## Struttura del codice

- `Measurement`: Classe per rappresentare una singola misurazione
- `MeasurementCard`: Widget personalizzato per visualizzare valori di misura
- `MetricsCard`: Widget per visualizzare metriche individuali
- `CableMeasurementApp`: Applicazione principale

## Personalizzazione

Puoi modificare i seguenti parametri nel codice:

- `base_diameter`: Diametro base del cavo da simulare (default: 10mm)
- `noise`: Livello di rumore nelle misurazioni (default: 5%)
- Frequenza di acquisizione: Modifica il valore in `acquisition_timer.start(100)` (in millisecondi)
- Densità del materiale: Modifica `density_steel` nella funzione `calculate_weight_per_meter`

## Licenza

Progetto basato sul design originale da Figma: Interfaccia Studio Fattibilità Cavo

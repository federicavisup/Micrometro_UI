# 🚀 Guida Rapida - Sistema di Misura Cavi

## Installazione Veloce

### Opzione 1: Setup Automatico
```bash
python setup.py
```

### Opzione 2: Installazione Manuale
```bash
pip install PyQt6 pyqtgraph numpy
```

## Avvio Applicazione

### Windows
Doppio click su `avvia_app.bat`

### Linux/Mac
```bash
./avvia_app.sh
```

Oppure:
```bash
python cable_measurement_app.py
```

## 📖 Utilizzo Base

### 1️⃣ Avviare l'Acquisizione
- Clicca sul pulsante **"▶ Avvia Acquisizione"**
- L'indicatore "⚡ In acquisizione" apparirà quando attivo
- Il grafico inizierà a mostrare i dati in tempo reale

### 2️⃣ Regolare la Finestra Temporale
- Usa lo **slider** per modificare la finestra di visualizzazione (1-10 secondi)
- Il numero di campioni si aggiorna automaticamente
- Lo slider è disabilitato durante l'acquisizione

### 3️⃣ Osservare le Misure
Il sistema mostra in tempo reale:
- **Dx Corrente**: Diametro misurato sull'asse X (blu)
- **Dy Corrente**: Diametro misurato sull'asse Y (arancione)  
- **Diametro Equivalente (ML)**: Stima calcolata dal modello (verde)
- **Peso per Metro**: Calcolato automaticamente (viola)

### 4️⃣ Confrontare con Valore Atteso
1. Inserisci il **valore atteso** in kg/m (es: 0.006157)
2. Clicca su **"Calcola Metriche"**
3. Visualizza le metriche di confronto:
   - ✅ **Bias**: Errore sistematico
   - 📊 **Errore %**: Errore percentuale
   - 📏 **Errore Assoluto**: Differenza assoluta
   - 🎯 **Accuratezza**: Precisione in percentuale
   - 📈 **RMSE**: Root Mean Square Error
   - 📉 **MAE**: Mean Absolute Error

### 5️⃣ Fermare l'Acquisizione
- Clicca su **"⏹ Arresta Acquisizione"**
- I dati rimangono visualizzati
- Puoi regolare nuovamente la finestra temporale

## 🎨 Interpretazione Colori

| Colore | Significato |
|--------|-------------|
| 🔵 Blu | Misure Dx (asse X) |
| 🟠 Arancione | Misure Dy (asse Y) |
| 🟢 Verde | Diametro equivalente ML |
| 🟣 Viola | Peso per metro |

### Badge Metriche
- ✅ **Verde**: Misura accettabile (errore < 5%)
- ⚠️ **Rosso**: Fuori tolleranza (errore ≥ 5%)

## 🔧 Parametri Simulazione

Il sistema simula attualmente:
- **Diametro base**: ~10 mm
- **Variazione**: ±5% (rumore realistico)
- **Frequenza acquisizione**: 10 Hz (100ms)
- **Densità acciaio**: 7850 kg/m³

## 📐 Formula Peso per Metro

```
Peso/metro = π × (D/2)² × ρ

Dove:
- D = diametro equivalente in mm
- ρ = 7850 kg/m³ (densità acciaio)
```

## 💡 Suggerimenti

1. **Stabilizzazione**: Attendi qualche secondo dopo l'avvio per permettere la stabilizzazione delle misure
2. **Finestra ottimale**: Una finestra di 5 secondi offre un buon bilanciamento tra reattività e stabilità
3. **Precisione**: Il sistema usa le ultime 10 misure per calcolare il diametro equivalente
4. **Confronto**: Inserisci valori attesi con almeno 6 decimali per un confronto accurato

## ❓ Risoluzione Problemi

### L'applicazione non si avvia
```bash
# Verifica l'installazione
python setup.py --check

# Reinstalla le dipendenze
pip install --force-reinstall PyQt6 pyqtgraph numpy
```

### Errori durante l'acquisizione
- Riavvia l'applicazione
- Verifica che non ci siano altre istanze in esecuzione

### Il grafico è vuoto
- Assicurati che l'acquisizione sia avviata (stato "In acquisizione")
- Controlla che la finestra temporale sia appropriata

## 📞 Supporto

Per problemi o domande:
1. Verifica i requisiti di sistema
2. Controlla i messaggi di errore nella console
3. Verifica la versione di Python (richiesto ≥ 3.8)

## 🔄 Aggiornamenti Futuri

Funzionalità pianificate:
- Esportazione dati in CSV
- Salvataggio report in JSON
- Configurazione parametri di simulazione
- Connessione a dispositivi reali di misura
- Grafici statistici avanzati

---

**Versione**: 1.0  
**Data**: Ottobre 2025  
**Basato su**: Interfaccia Studio Fattibilità Cavo (Figma)

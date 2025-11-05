# 🔧 Note Tecniche - Sviluppatori

## Architettura dell'Applicazione

### Struttura Classi

```
CableMeasurementApp (QMainWindow)
├── Measurement (Data class)
├── MeasurementCard (QGroupBox)
├── MetricsCard (QGroupBox)
└── PlotWidget (pyqtgraph)
```

### Flusso di Dati

```
acquire_data() [Timer 10Hz]
    ↓
Measurement object
    ↓
measurements deque (max 1000)
    ↓
estimate_equivalent_diameter()
    ↓
calculate_weight_per_meter()
    ↓
update_ui()
    ├── update cards
    ├── update chart
    └── update metrics
```

## Componenti Principali

### 1. Acquisizione Dati
```python
# Timer QT
self.acquisition_timer = QTimer()
self.acquisition_timer.timeout.connect(self.acquire_data)
self.acquisition_timer.start(100)  # 10 Hz
```

**Personalizzazione frequenza**:
- Modifica il valore in `start()` (millisecondi)
- Range consigliato: 50-500 ms (20 Hz - 2 Hz)

### 2. Gestione Memoria

```python
self.measurements: deque = deque(maxlen=1000)
```

- Usa `deque` per efficienza O(1) append/pop
- `maxlen=1000` limita memoria (circa 10 secondi a 10 Hz)
- Modifica `maxlen` per più/meno cronologia

### 3. Filtraggio Temporale

```python
cutoff_time = current_time - self.time_window
filtered = [m for m in measurements if m.timestamp > cutoff_time]
```

- Filtraggio lazy: solo quando necessario
- Non cancella dati, solo nasconde dal grafico
- Mantiene cronologia completa in `measurements`

### 4. Calcolo ML del Diametro

```python
def estimate_equivalent_diameter(dx: float, dy: float) -> float:
    geometric_mean = math.sqrt(dx * dy)
    ml_correction = 1.02 + (random.random() - 0.5) * 0.01
    return geometric_mean * ml_correction
```

**Personalizzazione**:
- `geometric_mean`: base matematica (media geometrica)
- `ml_correction`: fattore 1.02 ± 0.5% simula correzione ML
- Sostituisci con modello ML reale per produzione

### 5. Calcolo Peso

```python
def calculate_weight_per_meter(diameter: float) -> float:
    radius_m = (diameter / 2) / 1000  # mm → m
    area_m2 = math.pi * (radius_m ** 2)
    density_steel = 7850  # kg/m³
    return area_m2 * density_steel
```

**Materiali alternativi**:
```python
MATERIALS = {
    'acciaio': 7850,
    'alluminio': 2700,
    'rame': 8960,
    'ottone': 8500
}
```

## Grafici con pyqtgraph

### Configurazione Base
```python
self.plot_widget = PlotWidget()
self.plot_widget.setBackground('w')  # sfondo bianco
self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
```

### Curve Multiple
```python
self.dx_curve = self.plot_widget.plot(
    pen=pg.mkPen(color='#3b82f6', width=2),
    name='Dx (mm)'
)
```

### Update Performanti
```python
# Evita ricalcoli inutili
self.dx_curve.setData(times, dx_values)  # O(n) efficiente
```

## Stili PyQt6

### Metodo Consigliato: StyleSheets CSS-like

```python
widget.setStyleSheet("""
    QGroupBox {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }
""")
```

### Colori Standard (Tailwind-inspired)

```python
COLORS = {
    'primary': '#3b82f6',      # Blu
    'success': '#22c55e',      # Verde
    'warning': '#f59e0b',      # Arancione
    'danger': '#dc2626',       # Rosso
    'purple': '#9333ea',       # Viola
    'slate-50': '#f8fafc',     # Grigio chiaro
    'slate-600': '#64748b',    # Grigio medio
    'slate-900': '#0f172a'     # Grigio scuro
}
```

## Ottimizzazioni Performance

### 1. Grafici
```python
# Disabilita animazioni per real-time
isAnimationActive=False

# Disabilita punti per velocità
dot=False

# Limita punti visualizzati
max_points = 1000
if len(times) > max_points:
    times = times[-max_points:]
```

### 2. Timer
```python
# Evita accumulo eventi
self.acquisition_timer.setSingleShot(False)

# Priorità timer
self.acquisition_timer.setTimerType(Qt.TimerType.PreciseTimer)
```

### 3. Memoria
```python
# Usa deque invece di list
from collections import deque
measurements = deque(maxlen=1000)  # Auto-evict

# Evita copie inutili
filtered = [m for m in measurements if condition]  # OK
# invece di
filtered = measurements.copy()  # NO - spreco memoria
```

## Estensioni Possibili

### 1. Connessione Hardware Reale

```python
# Sostituisci acquire_data() con:
def acquire_data_from_sensor(self):
    # Leggi da seriale/USB/network
    dx, dy = self.sensor.read_measurements()
    
    # Gestisci errori
    if dx is None or dy is None:
        return
    
    # Prosegui come normale
    ...
```

### 2. Database Storage

```python
import sqlite3

def save_to_database(self, measurement):
    conn = sqlite3.connect('measurements.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO measurements (timestamp, dx, dy)
        VALUES (?, ?, ?)
    ''', (measurement.timestamp, measurement.dx, measurement.dy))
    conn.commit()
    conn.close()
```

### 3. Export Avanzato

```python
# Implementa nell'app principale
from export_module import export_measurements_to_csv

def export_data(self):
    filename = QFileDialog.getSaveFileName(...)
    if filename:
        export_measurements_to_csv(self.measurements, filename)
```

### 4. Configurazione Parametri

```python
class SettingsDialog(QDialog):
    def __init__(self):
        # Dialog per modificare:
        # - Diametro base simulazione
        # - Livello rumore
        # - Materiale cavo
        # - Frequenza acquisizione
        ...
```

### 5. Multi-Threading per Acquisizione

```python
from PyQt6.QtCore import QThread, pyqtSignal

class AcquisitionThread(QThread):
    data_ready = pyqtSignal(float, float)
    
    def run(self):
        while self.running:
            dx, dy = self.acquire()
            self.data_ready.emit(dx, dy)
            time.sleep(0.1)
```

## Debug e Testing

### Modalità Debug
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def acquire_data(self):
    logger.debug(f"Acquiring: dx={dx:.3f}, dy={dy:.3f}")
    ...
```

### Test Automatici
```python
import unittest

class TestMeasurements(unittest.TestCase):
    def test_diameter_calculation(self):
        dx, dy = 10.0, 10.0
        result = estimate_equivalent_diameter(dx, dy)
        self.assertAlmostEqual(result, 10.2, places=1)
```

## Requisiti Sistema

### Minimo
- Python 3.8+
- 4 GB RAM
- Qualsiasi OS (Windows/Linux/Mac)

### Consigliato
- Python 3.10+
- 8 GB RAM
- SSD per export veloci

## Dipendenze Versioni

```
PyQt6==6.6.1          # GUI framework
pyqtgraph==0.13.3     # Grafici real-time
numpy==1.26.2         # Calcoli numerici (indiretto)
```

**Note compatibilità**:
- PyQt6 ≥ 6.5.0 richiesto per features moderne
- pyqtgraph usa numpy internamente
- Testato su Windows 10/11, Ubuntu 22.04, macOS 13+

## Licenza & Crediti

Basato sul design Figma: "Interfaccia Studio Fattibilità Cavo"
- Framework: PyQt6 (GPL/Commercial)
- Grafici: pyqtgraph (MIT)

---

**Mantenuto da**: [Il tuo nome]  
**Versione**: 1.0  
**Ultimo aggiornamento**: Ottobre 2025

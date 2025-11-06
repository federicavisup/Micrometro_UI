"""
Sistema di Studio Fattibilità - Misura Diametro Cavi Metallici
Misurazione ortogonale Dx e Dy con stima ML del diametro equivalente

VERSIONE PYSIDE6 - Compatibile con Python 3.13
"""

import sys
import math
import random
from datetime import datetime
from collections import deque
from typing import List, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QGroupBox, QGridLayout,
    QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor

import pyqtgraph as pg
from pyqtgraph import PlotWidget

# new imports for prediction
from pathlib import Path
import tempfile
import pandas as pd
import os
import sys

# rendi importabile la cartella principale (Micrometro_UI) e importa predict.predict.predict_diameter se disponibile
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import predict



class Measurement:
    """Classe per rappresentare una singola misurazione"""
    def __init__(self, timestamp: float, dx: float, dy: float):
        self.timestamp = timestamp
        self.dx = dx
        self.dy = dy


class MeasurementCard(QGroupBox):
    """Widget personalizzato per mostrare una card con un valore di misura"""
    def __init__(self, title: str, description: str, color: str, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        
        layout = QVBoxLayout()
        
        # Label per la descrizione
        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(self.desc_label)
        
        # Label per il valore
        self.value_label = QLabel("0.000 mm")
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.value_label.setFont(font)
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #0f172a;
                font-weight: bold;
                font-size: 14px;
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)
    
    def set_value(self, value: float, unit: str = "mm"):
        """Aggiorna il valore visualizzato"""
        self.value_label.setText(f"{value:.3f} {unit}")


class MetricsCard(QGroupBox):
    """Widget per mostrare una metrica singola"""
    def __init__(self, title: str, unit: str, decimals: int = 3, highlight: bool = False, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Label per il titolo
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(title_label)
        
        # Label per il valore
        self.value_label = QLabel("0.000")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.value_label.setFont(font)
        if highlight:
            self.value_label.setStyleSheet("color: #15803d;")
        else:
            self.value_label.setStyleSheet("color: #0f172a;")
        layout.addWidget(self.value_label)
        
        # Label per l'unità
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #64748b; font-size: 10px;")
        layout.addWidget(unit_label)
        
        self.setLayout(layout)
        
        border_color = "#22c55e" if highlight else "#e2e8f0"
        bg_color = "#f0fdf4" if highlight else "white"
        
        self.setStyleSheet(f"""
            QGroupBox {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        
        self.decimals = decimals
    
    def set_value(self, value: float):
        """Aggiorna il valore visualizzato"""
        self.value_label.setText(f"{value:.{self.decimals}f}")


class CableMeasurementApp(QMainWindow):
    """Applicazione principale per la misurazione dei cavi"""
    
    def __init__(self):
        super().__init__()
        
        # Stato dell'applicazione
        self.is_acquiring = False
        self.measurements: deque = deque(maxlen=1000)  # Ultime 1000 misure
        self.current_dx = 0.0
        self.current_dy = 0.0
        # Non abbiamo ancora predizioni: usa None e visualizza "N/D" fino alla prima predizione
        self.equivalent_diameter = None  # mm or None
        self.weight_per_meter = None     # kg/m or None
        self.show_metrics = False

        # Predizione periodica: default 5s
        self.prediction_interval = 5  # seconds
        self.prediction_timer = QTimer()
        self.prediction_timer.timeout.connect(self.run_prediction)
        # temp folder to store csvs (optional)
        self._prediction_temp_dir = PROJECT_ROOT / 'code' / 'data' / 'temp'
        try:
            os.makedirs(self._prediction_temp_dir, exist_ok=True)
        except Exception:
            pass
        # end prediction setup

        # Analisi: finestra temporale (secondi) usata per preparare i dati per la predizione
        self.analysis_window_seconds = 5  # default (s)

        # Timer per l'acquisizione dati
        self.acquisition_timer = QTimer()
        self.acquisition_timer.timeout.connect(self.acquire_data)
        
        self.init_ui()
        # Forza visualizzazione iniziale (mostra N/D se non ci sono predizioni)
        self.update_ui()
        
    def init_ui(self):
        """Inizializza l'interfaccia utente"""
        self.setWindowTitle("Sistema di Studio Fattibilità - Misura Diametro Cavi Metallici")
        self.setGeometry(100, 100, 1400, 900)
        
        # Layout principale inserito in un content widget scrollabile
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_layout = QVBoxLayout()
        title = QLabel("Sistema di Studio Fattibilità - Misura Diametro Cavi Metallici")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #0f172a; margin: 10px;")
        
        subtitle = QLabel("Misurazione ortogonale Dx e Dy con stima ML del diametro equivalente")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #64748b; font-size: 13px; margin-bottom: 10px;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)
        
        # Controlli acquisizione
        controls_group = self.create_controls_section()
        main_layout.addWidget(controls_group)
        
        # Grafico
        chart_group = self.create_chart_section()
        main_layout.addWidget(chart_group)
        
        # Risultati
        results_layout = self.create_results_section()
        main_layout.addLayout(results_layout)
        
        # Peso per metro e confronto
        weight_group = self.create_weight_section()
        main_layout.addWidget(weight_group)
        
        # content_widget contiene il layout; viene inserito nella QScrollArea
        content_widget = QWidget()
        content_widget.setLayout(main_layout)
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Imposta la scroll area come central widget
        self.setCentralWidget(scroll)
         
         # Stile generale
        self.setStyleSheet("""
             QMainWindow {
                 background-color: #f8fafc;
             }
             QWidget {
                 background-color: #f8fafc;
             }
         """)
    
    def create_controls_section(self) -> QGroupBox:
        """Crea la sezione dei controlli"""
        group = QGroupBox("Controlli Acquisizione")
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                color: #0f172a;
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Descrizione
        desc = QLabel("Avvia/Arresta l'acquisizione delle misure")
        desc.setStyleSheet("color: #64748b; font-size: 12px; font-weight: normal; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Pulsante e status
        button_layout = QHBoxLayout()
        
        self.start_stop_btn = QPushButton("▶ Avvia Acquisizione")
        self.start_stop_btn.setMinimumWidth(200)
        self.start_stop_btn.setMinimumHeight(40)
        self.start_stop_btn.clicked.connect(self.toggle_acquisition)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        button_layout.addWidget(self.start_stop_btn)
        
        self.status_label = QLabel("")
        # Testo rosso per lo status (sfondo trasparente)
        self.status_label.setStyleSheet("""
            background-color: transparent;
            color: #dc2626;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
        """)
        self.status_label.hide()
        button_layout.addWidget(self.status_label)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Campo input: misura della finestra temporale analizzata (s)
        window_layout = QHBoxLayout()
        window_label = QLabel("Misura della finestra temporale analizzata (s):")
        window_label.setStyleSheet("color: #0f172a; font-size: 12px;")
        window_layout.addWidget(window_label)

        self.analysis_window_input = QLineEdit()
        self.analysis_window_input.setFixedWidth(80)
        self.analysis_window_input.setText(str(self.analysis_window_seconds))
        self.analysis_window_input.setToolTip("Inserire la finestra in secondi usata per costruire il file delle misure per la predizione")
        self.analysis_window_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 12px;
                background-color: white;
                color: #0f172a;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
                color: #0f172a;
            }
        """)
        window_layout.addWidget(self.analysis_window_input)
        window_layout.addStretch()
        layout.addLayout(window_layout)

        # Contatore campioni
        self.samples_label = QLabel("0 campioni")
        self.samples_label.setStyleSheet("color: #64748b; font-size: 12px;")
        button_layout.addWidget(self.samples_label)
        # (lo aggiungiamo anche nel layout principale per visibilità)
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_chart_section(self) -> QGroupBox:
        """Crea la sezione del grafico"""
        group = QGroupBox("Andamento Misure nel Tempo")
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                color: #0f172a;
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Descrizione
        desc = QLabel("Misure ortogonali Dx (blu) e Dy (arancione) in tempo reale")
        desc.setStyleSheet("color: #64748b; font-size: 12px; font-weight: normal; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Grafico
        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setMinimumHeight(350)
        self.plot_widget.setLabel('left', 'Diametro (mm)')
        self.plot_widget.setLabel('bottom', 'Tempo (s)')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        
        # Curve per Dx e Dy
        self.dx_curve = self.plot_widget.plot(pen=pg.mkPen(color='#3b82f6', width=2), name='Dx (mm)')
        self.dy_curve = self.plot_widget.plot(pen=pg.mkPen(color='#f97316', width=2), name='Dy (mm)')
        
        layout.addWidget(self.plot_widget)
        
        group.setLayout(layout)
        return group
    
    def create_results_section(self) -> QHBoxLayout:
        """Crea la sezione dei risultati"""
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        self.dx_card = MeasurementCard("Dx Corrente", "Diametro X", "#3b82f6")
        self.dy_card = MeasurementCard("Dy Corrente", "Diametro Y", "#f97316")
        self.equiv_card = MeasurementCard("Diametro Equivalente (ML)", "Stima modello ML", "#22c55e")
        
        layout.addWidget(self.dx_card)
        layout.addWidget(self.dy_card)
        layout.addWidget(self.equiv_card)
        
        return layout
    
    def create_weight_section(self) -> QGroupBox:
        """Crea la sezione del peso per metro"""
        group = QGroupBox("Peso per Metro Calcolato")
        group.setStyleSheet("""
            QGroupBox {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                color: #0f172a;
                subcontrol-origin: margin;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Descrizione
        desc = QLabel("Calcolato dalla formula: π × (D/2)² × ρ (ρ acciaio = 7850 kg/m³)")
        desc.setStyleSheet("color: #64748b; font-size: 12px; font-weight: normal; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Valore peso
        self.weight_label = QLabel("0.000000 kg/m")
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        self.weight_label.setFont(font)
        self.weight_label.setStyleSheet("color: #9333ea; margin: 10px 0;")
        layout.addWidget(self.weight_label)
        
        # Separatore
        separator = QLabel()
        separator.setStyleSheet("background-color: #e2e8f0; max-height: 1px;")
        separator.setMaximumHeight(1)
        layout.addWidget(separator)
        
        # Confronto
        compare_layout = QVBoxLayout()
        compare_layout.setSpacing(10)
        
        compare_title = QLabel("Confronto con Valore Atteso")
        compare_title.setStyleSheet("font-weight: bold; color: #0f172a; margin-top: 15px; font-size: 13px;")
        compare_layout.addWidget(compare_title)
        
        input_layout = QHBoxLayout()
        
        input_container = QVBoxLayout()
        input_label = QLabel("Valore Atteso (kg/m)")
        input_label.setStyleSheet("color: #0f172a; font-weight: bold; font-size: 11px;")
        input_container.addWidget(input_label)
        
        self.expected_input = QLineEdit()
        self.expected_input.setPlaceholderText("Es: 0.006157")
        # testo in blu per rendere il valore inserito subito visibile
        self.expected_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 13px;
                background-color: white;
                color: #3b82f6;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
            }
        """)
        input_container.addWidget(self.expected_input)
        input_layout.addLayout(input_container, 3)
        
        self.compare_btn = QPushButton("Calcola Metriche")
        self.compare_btn.setMinimumHeight(40)
        self.compare_btn.clicked.connect(self.calculate_metrics)
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #cbd5e1;
            }
        """)
        input_layout.addWidget(self.compare_btn, 1, Qt.AlignmentFlag.AlignBottom)
        
        compare_layout.addLayout(input_layout)
        
        # Contenitore metriche
        self.metrics_container = QWidget()
        self.metrics_layout = QVBoxLayout()
        self.metrics_container.setLayout(self.metrics_layout)
        self.metrics_container.hide()
        compare_layout.addWidget(self.metrics_container)
        
        layout.addLayout(compare_layout)
        
        group.setLayout(layout)
        return group
    
    def toggle_acquisition(self):
        """Avvia o ferma l'acquisizione"""
        if self.is_acquiring:
            # Ferma
            self.is_acquiring = False
            self.acquisition_timer.stop()
            # stop prediction timer if active
            if hasattr(self, 'prediction_timer') and self.prediction_timer.isActive():
                self.prediction_timer.stop()
            self.start_stop_btn.setText("▶ Avvia Acquisizione")
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
            self.status_label.hide()
        else:
            # Avvia
            self.is_acquiring = True
            self.measurements.clear()
            self.show_metrics = False
            self.metrics_container.hide()
            self.acquisition_timer.start(100)  # 10 Hz
            # start prediction timer if predictor available
            if predict.predict_diameter is None:
                # show warning but continue acquisition
                self.status_label.setText("⚠ predizione non disponibile")
                self.status_label.show()
            else:
                try:
                    self.prediction_timer.start(int(self.prediction_interval * 1000))
                except Exception:
                    pass
            # end start prediction
            self.start_stop_btn.setText("⏹ Arresta Acquisizione")
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc2626;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #b91c1c;
                }
            """)
            #self.status_label.setText("⚡ In acquisizione")
            #self.status_label.show()
    
    def acquire_data(self):
        """Simula l'acquisizione di dati"""
        # Simula misure di un cavo da circa 10mm con variazioni realistiche
        base_diameter = 11.0  # mm
        noise = 0.05  # variazione del 5%
        
        dx = base_diameter + (random.random() - 0.5) * base_diameter * noise
        dy = base_diameter + (random.random() - 0.5) * base_diameter * noise
        
        self.current_dx = dx
        self.current_dy = dy
        
        # Crea nuova misurazione
        timestamp = datetime.now().timestamp()
        measurement = Measurement(timestamp, dx, dy)
        self.measurements.append(measurement)
        
        # NOTA: non eseguiamo più una stima fittizia locale del diametro.
        # La variabile self.equivalent_diameter verrà aggiornata solamente
        # quando run_prediction() esegue la predizione reale (ogni prediction_interval secondi).
        # Quindi qui manteniamo solo la simulazione dei sensori (Dx/Dy).
        # Aggiorniamo il peso solo se abbiamo già una predizione valida.
        if self.equivalent_diameter is not None:
            self.weight_per_meter = self.calculate_weight_per_meter(self.equivalent_diameter)
        else:
            self.weight_per_meter = None
         
         # Aggiorna UI
        self.update_ui()
    
    # ...removed fake ML estimator...
    # La stima ML ora viene eseguita solo dalla funzione run_prediction()
    # che chiama il modulo di predizione reale in code/predict.py.

    def run_prediction(self):
        """Esegue predizione periodica: crea CSV temporaneo da misure recenti e chiama predict.predict.predict_diameter"""
        if predict.predict_diameter is None:
            return
        try:
            # determina la finestra temporale da input (in secondi)
            try:
                secs = float(self.analysis_window_input.text())
                if secs <= 0:
                    secs = self.analysis_window_seconds
            except Exception:
                secs = self.analysis_window_seconds
            # seleziona le misure più recenti entro la finestra secs
            now_ts = datetime.now().timestamp()
            cutoff = now_ts - secs
            samples = [m for m in self.measurements if m.timestamp >= cutoff]
            # fallback: se non ci sono misure nella finestra, usa gli ultimi 200 come riserva
            if len(samples) == 0:
                samples = list(self.measurements)[-200:]
            if len(samples) == 0:
                 self.status_label.setText("⚠ nessuna misura per predizione")
                 self.status_label.show()
                 return

            # build dataframe with Tempo,Dx,Dy (Tempo formatted as HH:MM:SS.mmm)
            tempos = []
            dxs = []
            dys = []
            for m in samples:
                # formatta il timestamp in ora:min:sec.milliseconds (es. 15:05:24.122)
                try:
                    dt = datetime.fromtimestamp(m.timestamp)
                    tempo_str = dt.strftime("%H:%M:%S.%f")[:-3]  # togliere ultime 3 cifre microsecondi -> millisecondi
                except Exception:
                    tempo_str = ""
                tempos.append(tempo_str)
                dxs.append(m.dx)
                dys.append(m.dy)

            df = pd.DataFrame({
                'Tempo': tempos,
                'Dx': dxs,
                'Dy': dys
            })

            # write to a temporary csv file inside temp dir (or system temp)
            tmp_f = None
            try:
                tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', dir=str(self._prediction_temp_dir))
                tmp_f = tmp.name
                df.to_csv(tmp_f, index=False)
                tmp.close()
            except Exception:
                # fallback to system temp
                tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
                tmp_f = tmp.name
                df.to_csv(tmp_f, index=False)
                tmp.close()

            # call predict.predict.predict_diameter with absolute path; model_dir in project code/models
            try:
                model_dir = str(PROJECT_ROOT / 'code' / 'models')
                res = predict.predict_diameter(filename=str(tmp_f), data_folder="", model_dir=model_dir, verbose=False)
            except Exception as e:
                # show brief error and cleanup
                self.status_label.setText("❌ Errore predizione")
                self.status_label.show()
                try:
                    os.unlink(tmp_f)
                except Exception:
                    pass
                return

            # cleanup temp file
            try:
                os.unlink(tmp_f)
            except Exception:
                pass

            # get estimated diameter and update UI
            estimated = res.get('estimated_diameter', None)
            if estimated is not None:
                # Aggiorniamo il diametro equivalente con il valore restituito dal modello.
                # Se il tuo modello restituisce unità diverse (es. decimi di mm) modifica qui la conversione.
                self.equivalent_diameter = float(estimated)/10
                self.weight_per_meter = self.calculate_weight_per_meter(self.equivalent_diameter)
                # update status and UI
                self.status_label.setText(f"✓ Pred: {self.equivalent_diameter:.3f} mm (win={secs:.1f}s)")
                self.status_label.show()
                self.update_ui()
        except Exception:
            # do not stop timer on unexpected errors
            self.status_label.setText("❌ Errore predizione")
            self.status_label.show()
            return

    def update_ui(self):
        """Aggiorna l'interfaccia con i nuovi valori"""
        # Aggiorna cards
        self.dx_card.set_value(self.current_dx)
        self.dy_card.set_value(self.current_dy)
        # Diametro equivalente: mostra N/D se non disponibile
        if self.equivalent_diameter is None:
            self.equiv_card.value_label.setText("N/D")
        else:
            self.equiv_card.set_value(self.equivalent_diameter)

        # Peso per metro: mostra N/D se non disponibile
        if self.weight_per_meter is None:
            self.weight_label.setText("N/D")
        else:
            self.weight_label.setText(f"{self.weight_per_meter:.6f} kg/m")
        
        # Aggiorna contatore campioni
        self.samples_label.setText(f"{len(self.measurements)} campioni")
        
        # Aggiorna grafico
        self.update_chart()

    def update_chart(self):
        """Aggiorna il grafico con i dati presenti in self.measurements"""
        # Verifiche rapide
        if not hasattr(self, 'dx_curve') or not hasattr(self, 'dy_curve') or not hasattr(self, 'plot_widget'):
            return
        if not self.measurements:
            # svuota le curve se non ci sono dati
            try:
                self.dx_curve.setData([], [])
                self.dy_curve.setData([], [])
            except Exception:
                pass
            return

        # Usa tutte le misure nella deque (maxlen gestito dalla deque)
        filtered = list(self.measurements)
        start_time = filtered[0].timestamp
        times = [(m.timestamp - start_time) for m in filtered]
        dx_values = [m.dx for m in filtered]
        dy_values = [m.dy for m in filtered]

        # Aggiorna le curve
        try:
            self.dx_curve.setData(times, dx_values)
            self.dy_curve.setData(times, dy_values)
            # Adatta l'asse X ai dati (se ci sono almeno 2 punti)
            if len(times) >= 2:
                xmin, xmax = min(times), max(times)
                # lascia un piccolo padding
                padding = max(0.1, (xmax - xmin) * 0.02)
                self.plot_widget.setXRange(xmin - padding, xmax + padding, padding=0)
        except Exception:
            # non sollevare errori di plotting in runtime
            pass

    def calculate_weight_per_meter(self, diameter: float) -> float:
        """Calcola il peso per metro (kg/m) di un cavo in acciaio dato il diametro in mm."""
        if diameter is None:
            return None
        # converti diametro mm -> raggio in metri
        radius_m = (diameter/ 2.0) / 1000.0
        area_m2 = math.pi * (radius_m ** 2)
        density_steel = 7850.0  # kg/m³
        return area_m2 * density_steel

    def calculate_metrics(self):
        """Calcola e visualizza le metriche di confronto"""
        try:
            expected = float(self.expected_input.text())
            if expected <= 0 or self.weight_per_meter <= 0:
                return
            
            # Rimuovi widget precedenti
            while self.metrics_layout.count():
                child = self.metrics_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            # Calcola metriche
            error = self.weight_per_meter - expected
            percentage_error = (error / expected) * 100
            absolute_error = abs(error)
            relative_error = abs(percentage_error)
            bias = error
            accuracy = 100 - relative_error
            rmse = absolute_error
            mae = absolute_error
            
            is_acceptable = relative_error < 5
            
            # Contenitore con sfondo
            container = QWidget()
            container.setStyleSheet("""
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 15px;
            """)
            container_layout = QVBoxLayout()
            
            # Header
            header_layout = QHBoxLayout()
            header_title = QLabel("Metriche di Confronto")
            header_title.setStyleSheet("font-weight: bold; color: #0f172a; font-size: 13px;")
            header_layout.addWidget(header_title)
            
            status_badge = QLabel("✓ Accettabile" if is_acceptable else "⚠ Fuori tolleranza")
            status_color = "#22c55e" if is_acceptable else "#dc2626"
            status_badge.setStyleSheet(f"""
                background-color: {status_color};
                color: white;
                padding: 4px 10px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 11px;
            """)
            header_layout.addWidget(status_badge)
            header_layout.addStretch()
            
            container_layout.addLayout(header_layout)
            
            # Griglia metriche principali
            metrics_grid = QGridLayout()
            metrics_grid.setSpacing(10)
            
            bias_card = MetricsCard("Bias", "kg/m", 6)
            bias_card.set_value(bias)
            metrics_grid.addWidget(bias_card, 0, 0)
            
            error_pct_card = MetricsCard("Errore %", "%", 3)
            error_pct_card.set_value(percentage_error)
            metrics_grid.addWidget(error_pct_card, 0, 1)
            
            abs_error_card = MetricsCard("Errore Assoluto", "kg/m", 6)
            abs_error_card.set_value(absolute_error)
            metrics_grid.addWidget(abs_error_card, 0, 2)
            
            accuracy_card = MetricsCard("Accuratezza", "%", 2, highlight=True)
            accuracy_card.set_value(accuracy)
            metrics_grid.addWidget(accuracy_card, 0, 3)
            
            container_layout.addLayout(metrics_grid)
            
            # Griglia metriche secondarie
            secondary_grid = QGridLayout()
            secondary_grid.setSpacing(10)
            
            rmse_card = MetricsCard("RMSE", "kg/m", 6)
            rmse_card.set_value(rmse)
            secondary_grid.addWidget(rmse_card, 0, 0)
            
            mae_card = MetricsCard("MAE", "kg/m", 6)
            mae_card.set_value(mae)
            secondary_grid.addWidget(mae_card, 0, 1)
            
            container_layout.addLayout(secondary_grid)
            
            # Valori confronto
            values_layout = QGridLayout()
            values_layout.setSpacing(15)
            
            measured_container = QVBoxLayout()
            measured_label = QLabel("Valore Misurato:")
            measured_label.setStyleSheet("color: #64748b; font-size: 11px;")
            measured_value = QLabel(f"{self.weight_per_meter:.6f} kg/m")
            measured_value.setStyleSheet("color: #9333ea; font-weight: bold; font-size: 13px;")
            measured_container.addWidget(measured_label)
            measured_container.addWidget(measured_value)
            values_layout.addLayout(measured_container, 0, 0)
            
            expected_container = QVBoxLayout()
            expected_label = QLabel("Valore Atteso:")
            expected_label.setStyleSheet("color: #64748b; font-size: 11px;")
            expected_container.addWidget(expected_label)
            expected_value = QLabel(f"{expected:.6f} kg/m")
            # testo in blu per evidenziare il valore atteso
            expected_value.setStyleSheet("color: #3b82f6; font-weight: bold; font-size: 13px;")
            expected_container.addWidget(expected_value)
            values_layout.addLayout(expected_container, 0, 1)
            
            container_layout.addLayout(values_layout)
            
            container.setLayout(container_layout)
            self.metrics_layout.addWidget(container)
            self.metrics_container.show()
            
        except ValueError:
            pass


def main():
    """Funzione principale"""
    app = QApplication(sys.argv)
    
    # Imposta lo stile dell'applicazione
    app.setStyle('Fusion')
    
    # Palette personalizzata
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(248, 250, 252))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(15, 23, 42))
    app.setPalette(palette)
    
    window = CableMeasurementApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
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
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor

import pyqtgraph as pg
from pyqtgraph import PlotWidget


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
        self.time_window = 5  # secondi
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.equivalent_diameter = 0.0
        self.weight_per_meter = 0.0
        self.show_metrics = False
        
        # Timer per l'acquisizione dati
        self.acquisition_timer = QTimer()
        self.acquisition_timer.timeout.connect(self.acquire_data)
        
        self.init_ui()
        
    def init_ui(self):
        """Inizializza l'interfaccia utente"""
        self.setWindowTitle("Sistema di Studio Fattibilità - Misura Diametro Cavi Metallici")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principale
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
        
        central_widget.setLayout(main_layout)
        
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
        desc = QLabel("Avvia/Arresta l'acquisizione delle misure e regola la finestra temporale")
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
        self.status_label.setStyleSheet("""
            background-color: #3b82f6;
            color: white;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
        """)
        self.status_label.hide()
        button_layout.addWidget(self.status_label)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Slider finestra temporale
        slider_layout = QVBoxLayout()
        slider_layout.setSpacing(5)
        
        slider_info = QHBoxLayout()
        self.time_window_label = QLabel(f"Finestra Temporale: {self.time_window}s")
        self.time_window_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #0f172a;")
        slider_info.addWidget(self.time_window_label)
        
        self.samples_label = QLabel("0 campioni")
        self.samples_label.setStyleSheet("color: #64748b; font-size: 12px;")
        slider_info.addStretch()
        slider_info.addWidget(self.samples_label)
        
        slider_layout.addLayout(slider_info)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(10)
        self.time_slider.setValue(self.time_window)
        self.time_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.time_slider.setTickInterval(1)
        self.time_slider.valueChanged.connect(self.on_time_window_changed)
        self.time_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: none;
                width: 18px;
                height: 18px;
                border-radius: 9px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #2563eb;
            }
        """)
        
        slider_layout.addWidget(self.time_slider)
        layout.addLayout(slider_layout)
        
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
        self.expected_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 13px;
                background-color: white;
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
            self.time_slider.setEnabled(True)
        else:
            # Avvia
            self.is_acquiring = True
            self.measurements.clear()
            self.show_metrics = False
            self.metrics_container.hide()
            self.acquisition_timer.start(100)  # 10 Hz
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
            self.status_label.setText("⚡ In acquisizione")
            self.status_label.show()
            self.time_slider.setEnabled(False)
    
    def on_time_window_changed(self, value: int):
        """Gestisce il cambio della finestra temporale"""
        self.time_window = value
        self.time_window_label.setText(f"Finestra Temporale: {self.time_window}s")
    
    def acquire_data(self):
        """Simula l'acquisizione di dati"""
        # Simula misure di un cavo da circa 10mm con variazioni realistiche
        base_diameter = 10.0  # mm
        noise = 0.05  # variazione del 5%
        
        dx = base_diameter + (random.random() - 0.5) * base_diameter * noise
        dy = base_diameter + (random.random() - 0.5) * base_diameter * noise
        
        self.current_dx = dx
        self.current_dy = dy
        
        # Crea nuova misurazione
        timestamp = datetime.now().timestamp()
        measurement = Measurement(timestamp, dx, dy)
        self.measurements.append(measurement)
        
        # Filtra misure nella finestra temporale
        cutoff_time = timestamp - self.time_window
        filtered_measurements = [m for m in self.measurements if m.timestamp > cutoff_time]
        
        # Calcola diametro equivalente
        if len(filtered_measurements) >= 10:
            recent = list(filtered_measurements)[-10:]
            avg_dx = sum(m.dx for m in recent) / len(recent)
            avg_dy = sum(m.dy for m in recent) / len(recent)
            self.equivalent_diameter = self.estimate_equivalent_diameter(avg_dx, avg_dy)
        else:
            self.equivalent_diameter = self.estimate_equivalent_diameter(dx, dy)
        
        self.weight_per_meter = self.calculate_weight_per_meter(self.equivalent_diameter)
        
        # Aggiorna UI
        self.update_ui()
    
    def estimate_equivalent_diameter(self, dx: float, dy: float) -> float:
        """Stima il diametro equivalente con correzione ML"""
        geometric_mean = math.sqrt(dx * dy)
        ml_correction = 1.02 + (random.random() - 0.5) * 0.01
        return geometric_mean * ml_correction
    
    def calculate_weight_per_meter(self, diameter: float) -> float:
        """Calcola il peso per metro di un cavo in acciaio"""
        radius_m = (diameter / 2) / 1000  # da mm a m
        area_m2 = math.pi * (radius_m ** 2)
        density_steel = 7850  # kg/m³
        return area_m2 * density_steel
    
    def update_ui(self):
        """Aggiorna l'interfaccia con i nuovi valori"""
        # Aggiorna cards
        self.dx_card.set_value(self.current_dx)
        self.dy_card.set_value(self.current_dy)
        self.equiv_card.set_value(self.equivalent_diameter)
        
        # Aggiorna peso
        self.weight_label.setText(f"{self.weight_per_meter:.6f} kg/m")
        
        # Aggiorna contatore campioni
        self.samples_label.setText(f"{len(self.measurements)} campioni")
        
        # Aggiorna grafico
        self.update_chart()
    
    def update_chart(self):
        """Aggiorna il grafico con i dati recenti"""
        if not self.measurements:
            return
        
        # Filtra misure nella finestra temporale
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - self.time_window
        filtered = [m for m in self.measurements if m.timestamp > cutoff_time]
        
        if not filtered:
            return
        
        # Prepara dati per il grafico
        start_time = filtered[0].timestamp
        times = [(m.timestamp - start_time) for m in filtered]
        dx_values = [m.dx for m in filtered]
        dy_values = [m.dy for m in filtered]
        
        # Aggiorna curve
        self.dx_curve.setData(times, dx_values)
        self.dy_curve.setData(times, dy_values)
    
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
            expected_value = QLabel(f"{expected:.6f} kg/m")
            expected_value.setStyleSheet("color: #0f172a; font-weight: bold; font-size: 13px;")
            expected_container.addWidget(expected_label)
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
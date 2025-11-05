"""
Versione estesa con funzionalità di esportazione dati
"""

import csv
from datetime import datetime
from PyQt6.QtWidgets import QFileDialog, QMessageBox
import json

def export_measurements_to_csv(measurements, filename):
    """Esporta le misurazioni in formato CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Data/Ora', 'Dx (mm)', 'Dy (mm)'])
        
        for m in measurements:
            dt = datetime.fromtimestamp(m.timestamp)
            writer.writerow([
                m.timestamp,
                dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                f"{m.dx:.6f}",
                f"{m.dy:.6f}"
            ])

def export_report(data, filename):
    """Esporta un report completo in JSON"""
    report = {
        'data_acquisizione': datetime.now().isoformat(),
        'configurazione': {
            'finestra_temporale_s': data['time_window'],
            'numero_campioni': data['num_samples'],
            'frequenza_acquisizione_hz': 10
        },
        'misurazioni': {
            'dx_medio_mm': data['avg_dx'],
            'dy_medio_mm': data['avg_dy'],
            'diametro_equivalente_mm': data['equivalent_diameter'],
            'peso_per_metro_kg': data['weight_per_meter']
        },
        'confronto': data.get('metrics', {})
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

# Aggiungi questi metodi alla classe CableMeasurementApp:

def add_export_buttons(self):
    """Aggiunge pulsanti per l'esportazione (da inserire nel create_controls_section)"""
    export_layout = QHBoxLayout()
    
    export_csv_btn = QPushButton("📊 Esporta CSV")
    export_csv_btn.clicked.connect(self.export_to_csv)
    export_csv_btn.setStyleSheet("""
        QPushButton {
            background-color: #059669;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #047857;
        }
    """)
    
    export_report_btn = QPushButton("📄 Esporta Report")
    export_report_btn.clicked.connect(self.export_report)
    export_report_btn.setStyleSheet("""
        QPushButton {
            background-color: #7c3aed;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #6d28d9;
        }
    """)
    
    export_layout.addWidget(export_csv_btn)
    export_layout.addWidget(export_report_btn)
    export_layout.addStretch()
    
    return export_layout

def export_to_csv(self):
    """Esporta le misurazioni in CSV"""
    if not self.measurements:
        QMessageBox.warning(self, "Nessun dato", "Non ci sono misurazioni da esportare.")
        return
    
    filename, _ = QFileDialog.getSaveFileName(
        self,
        "Salva misurazioni",
        f"misurazioni_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "CSV Files (*.csv)"
    )
    
    if filename:
        try:
            export_measurements_to_csv(self.measurements, filename)
            QMessageBox.information(self, "Successo", f"Dati esportati in:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore durante l'esportazione:\n{str(e)}")

def export_report(self):
    """Esporta un report completo in JSON"""
    if not self.measurements:
        QMessageBox.warning(self, "Nessun dato", "Non ci sono misurazioni da esportare.")
        return
    
    # Calcola medie
    measurements_list = list(self.measurements)
    avg_dx = sum(m.dx for m in measurements_list) / len(measurements_list)
    avg_dy = sum(m.dy for m in measurements_list) / len(measurements_list)
    
    data = {
        'time_window': self.time_window,
        'num_samples': len(self.measurements),
        'avg_dx': avg_dx,
        'avg_dy': avg_dy,
        'equivalent_diameter': self.equivalent_diameter,
        'weight_per_meter': self.weight_per_meter
    }
    
    # Aggiungi metriche se disponibili
    if self.show_metrics and self.expected_input.text():
        try:
            expected = float(self.expected_input.text())
            error = self.weight_per_meter - expected
            percentage_error = (error / expected) * 100
            
            data['metrics'] = {
                'valore_atteso_kg_m': expected,
                'bias_kg_m': error,
                'errore_percentuale': percentage_error,
                'errore_assoluto_kg_m': abs(error),
                'accuratezza_percentuale': 100 - abs(percentage_error)
            }
        except:
            pass
    
    filename, _ = QFileDialog.getSaveFileName(
        self,
        "Salva report",
        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "JSON Files (*.json)"
    )
    
    if filename:
        try:
            export_report(data, filename)
            QMessageBox.information(self, "Successo", f"Report esportato in:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore durante l'esportazione:\n{str(e)}")

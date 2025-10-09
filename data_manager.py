#!/usr/bin/env python3
"""
Data Manager per gestire molti file di dati dei tondini
Autore: Assistente AI
Data: 2025

Questo script gestisce automaticamente la lettura e organizzazione 
di molti file CSV con dati di tondini.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple
import glob

class TondinoDataManager:
    def __init__(self, data_dir="data"):
        """
        Inizializza il data manager
        
        Args:
            data_dir: directory contenente i dati
        """
        self.data_dir = Path(data_dir)
        self.config = {}
        self.file_registry = []
        
    def auto_detect_files(self, pattern="*.csv"):
        """
        Rileva automaticamente i file CSV e cerca di estrarre informazioni dal nome
        """
        print(f"🔍 Rilevamento automatico file in {self.data_dir}...")
        
        # Cerca file ricorsivamente
        csv_files = list(self.data_dir.rglob(pattern))
        
        detected_files = []
        
        for file_path in csv_files:
            file_info = self._extract_info_from_filename(file_path)
            if file_info:
                detected_files.append(file_info)
                print(f"   ✅ {file_path.name} -> D{file_info['diameter']}mm {file_info['type']}")
            else:
                print(f"   ❓ {file_path.name} -> Info non rilevabile automaticamente")
        
        self.file_registry = detected_files
        print(f"📊 Rilevati {len(detected_files)} file con info automatiche")
        
        return detected_files
    
    def _extract_info_from_filename(self, file_path):
        """
        Estrae informazioni dal nome del file usando pattern comuni
        """
        filename = file_path.name.lower()
        
        # Pattern per rilevare diametro
        diameter_patterns = [
            r'd(\d+)_',           # D8_
            r'_d(\d+)_',          # _D8_
            r'(\d+)mm',           # 8mm
            r'diam(\d+)',         # diam8
            r'phi(\d+)',          # phi8
            
        ]
        
        diameter = None
        for pattern in diameter_patterns:
            match = re.search(pattern, filename)
            if match:
                diameter = float(match.group(1))
                break
        
        # Pattern per rilevare tipo (liscio/nervato)
        bar_type = 'unknown'
        if any(word in filename for word in ['liscio', 'lisce', 'smooth', 'plain']):
            bar_type = 'liscio'
        elif any(word in filename for word in ['nervato', 'creste', 'ribbed', 'deformed']):
            bar_type = 'nervato'
        
        if diameter:
            return {
                'file_path': str(file_path),
                'diameter': diameter,
                'type': bar_type,
                'filename': file_path.name
            }
        
        return None
    
    def create_config_from_detection(self, output_file="data_config.json"):
        """
        Crea un file di configurazione basato sul rilevamento automatico
        """
        if not self.file_registry:
            print("⚠️  Nessun file rilevato. Esegui prima auto_detect_files()")
            return False
        
        # Organizza per diametro e tipo
        config = {}
        
        for file_info in self.file_registry:
            diameter = file_info['diameter']
            bar_type = file_info['type']
            
            key = f"D{diameter}_{bar_type}"
            
            if key not in config:
                config[key] = {
                    'diameter': diameter,
                    'type': bar_type,
                    'files': []
                }
            
            config[key]['files'].append({
                'path': file_info['file_path'],
                'filename': file_info['filename']
            })
        
        # Salva configurazione
        config_path = self.data_dir / output_file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.config = config
        
        print(f"📄 Configurazione salvata in {config_path}")
        self._print_config_summary()
        
        return True
    
    def load_config(self, config_file="data_config.json"):
        """
        Carica configurazione da file JSON
        """
        config_path = self.data_dir / config_file
        
        if not config_path.exists():
            print(f"❌ File configurazione non trovato: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            print(f"✅ Configurazione caricata da {config_path}")
            self._print_config_summary()
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento configurazione: {e}")
            return False
    
    def _print_config_summary(self):
        """
        Stampa un riassunto della configurazione
        """
        print("\n📋 RIASSUNTO CONFIGURAZIONE:")
        print("-" * 40)
        
        total_files = 0
        for key, info in self.config.items():
            n_files = len(info['files'])
            total_files += n_files
            print(f"{key}: {n_files} file(s) - D{info['diameter']}mm {info['type']}")
        
        print("-" * 40)
        print(f"TOTALE: {total_files} file(s) in {len(self.config)} categorie")
        print()
    
    def get_file_list_for_training(self):
        """
        Restituisce la lista di file nel formato richiesto dal training script
        
        Returns:
            List[Tuple]: Lista di (file_path, diameter, type)
        """
        if not self.config:
            print("❌ Nessuna configurazione caricata")
            return []
        
        file_list = []
        
        for key, info in self.config.items():
            diameter = info['diameter']
            bar_type = info['type']
            
            for file_info in info['files']:
                file_path = file_info['path']
                
                # Verifica che il file esista
                if os.path.exists(file_path):
                    file_list.append((file_path, diameter, bar_type))
                else:
                    print(f"⚠️  File non trovato: {file_path}")
        
        print(f"📤 Preparati {len(file_list)} file per il training")
        return file_list
    
    def validate_files(self):
        """
        Valida tutti i file nella configurazione
        """
        print("🔍 Validazione file...")
        
        valid_files = 0
        invalid_files = 0
        issues = []
        
        for key, info in self.config.items():
            for file_info in info['files']:
                file_path = file_info['path']
                
                try:
                    # Tenta di leggere il file
                    df = pd.read_csv(file_path)
                    
                    # Verifica colonne necessarie
                    required_cols = ['Tempo', 'Dx', 'Dy']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        issues.append(f"{file_path}: Colonne mancanti: {missing_cols}")
                        invalid_files += 1
                        continue
                    
                    # Verifica dati numerici
                    if df['Dx'].dtype not in [np.float64, np.int64] or df['Dy'].dtype not in [np.float64, np.int64]:
                        try:
                            df['Dx'] = pd.to_numeric(df['Dx'], errors='coerce')
                            df['Dy'] = pd.to_numeric(df['Dy'], errors='coerce')
                        except:
                            issues.append(f"{file_path}: Dati Dx/Dy non numerici")
                            invalid_files += 1
                            continue
                    
                    # Verifica numero di righe ragionevole
                    if len(df) < 10:
                        issues.append(f"{file_path}: Troppo pochi dati ({len(df)} righe)")
                        invalid_files += 1
                        continue
                    
                    valid_files += 1
                    
                except Exception as e:
                    issues.append(f"{file_path}: Errore lettura: {e}")
                    invalid_files += 1
        
        # Riassunto validazione
        print(f"\n📊 RISULTATI VALIDAZIONE:")
        print(f"✅ File validi: {valid_files}")
        print(f"❌ File invalidi: {invalid_files}")
        
        if issues:
            print(f"\n⚠️  PROBLEMI RILEVATI:")
            for issue in issues[:10]:  # Mostra solo i primi 10
                print(f"   {issue}")
            if len(issues) > 10:
                print(f"   ... e altri {len(issues) - 10} problemi")
        
        return valid_files, invalid_files, issues
    
    def create_sample_config(self):
        """
        Crea un file di configurazione di esempio
        """
        sample_config = {
            "D6_nervato": {
                "diameter": 6,
                "type": "nervato",
                "files": [
                    {"path": "data/D6_nervati/D6_creste_1.csv", "filename": "D6_creste_1.csv"},
                    {"path": "data/D6_nervati/D6_creste_2.csv", "filename": "D6_creste_2.csv"}
                ]
            },
            "D8_liscio": {
                "diameter": 8,
                "type": "liscio",
                "files": [
                    {"path": "data/D8_lisci/D8_liscio_1.csv", "filename": "D8_liscio_1.csv"},
                    {"path": "data/D8_lisci/D8_liscio_2.csv", "filename": "D8_liscio_2.csv"}
                ]
            },
            "D8_nervato": {
                "diameter": 8,
                "type": "nervato",
                "files": [
                    {"path": "data/D8_nervati/D8_creste_1.csv", "filename": "D8_creste_1.csv"},
                    {"path": "data/D8_nervati/D8_creste_2.csv", "filename": "D8_creste_2.csv"}
                ]
            }
        }
        
        sample_path = self.data_dir / "sample_config.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Configurazione di esempio creata: {sample_path}")
        print("💡 Modifica questo file per adattarlo ai tuoi dati")
    
    def interactive_config_creation(self):
        """
        Creazione interattiva della configurazione
        """
        print("\n🎛️  CREAZIONE CONFIGURAZIONE INTERATTIVA")
        print("=" * 50)
        
        config = {}
        
        while True:
            print(f"\nConfigurazione attuale: {len(config)} categorie")
            
            action = input("\nCosa vuoi fare?\n1) Aggiungi categoria\n2) Mostra configurazione\n3) Salva e termina\n4) Annulla\nScelta: ").strip()
            
            if action == "1":
                self._add_category_interactive(config)
            elif action == "2":
                self._show_config_interactive(config)
            elif action == "3":
                if config:
                    self.config = config
                    config_path = self.data_dir / "interactive_config.json"
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"✅ Configurazione salvata: {config_path}")
                    return True
                else:
                    print("❌ Configurazione vuota!")
            elif action == "4":
                print("❌ Operazione annullata")
                return False
            else:
                print("❌ Scelta non valida")
    
    def _add_category_interactive(self, config):
        """Aggiunge una categoria interattivamente"""
        try:
            diameter = float(input("Diametro (mm): "))
            bar_type = input("Tipo (liscio/nervato): ").strip().lower()
            
            if bar_type not in ['liscio', 'nervato']:
                print("❌ Tipo deve essere 'liscio' o 'nervato'")
                return
            
            key = f"D{diameter}_{bar_type}"
            
            if key in config:
                print(f"⚠️  Categoria {key} già esiste")
                return
            
            # Aggiungi file
            files = []
            while True:
                file_path = input(f"File path per {key} (vuoto per terminare): ").strip()
                if not file_path:
                    break
                
                if os.path.exists(file_path):
                    files.append({
                        "path": file_path,
                        "filename": os.path.basename(file_path)
                    })
                    print(f"   ✅ Aggiunto: {file_path}")
                else:
                    print(f"   ❌ File non trovato: {file_path}")
            
            if files:
                config[key] = {
                    "diameter": diameter,
                    "type": bar_type,
                    "files": files
                }
                print(f"✅ Categoria {key} creata con {len(files)} file(s)")
            else:
                print("❌ Nessun file valido aggiunto")
                
        except ValueError:
            print("❌ Diametro deve essere un numero intero")
    
    def _show_config_interactive(self, config):
        """Mostra la configurazione attuale"""
        if not config:
            print("📭 Configurazione vuota")
            return
        
        print("\n📋 CONFIGURAZIONE ATTUALE:")
        print("-" * 40)
        for key, info in config.items():
            print(f"{key}: {len(info['files'])} file(s)")
            for file_info in info['files']:
                print(f"   - {file_info['filename']}")


def main():
    """
    Esempio di utilizzo del data manager
    """
    print("🎯 TONDINO DATA MANAGER")
    print("=" * 30)
    
    # Crea data manager
    dm = TondinoDataManager("data")
    
    # Crea directory se non esiste
    dm.data_dir.mkdir(exist_ok=True)
    
    print("\nOpzioni disponibili:")
    print("1) Auto-rileva file CSV")
    print("2) Carica configurazione esistente")
    print("3) Crea configurazione interattiva")
    print("4) Crea configurazione di esempio")
    print("5) Valida file esistenti")
    
    choice = input("\nScelta: ").strip()
    
    if choice == "1":
        # Auto rilevamento
        detected = dm.auto_detect_files()
        if detected:
            if input("\nCreare configurazione da rilevamento automatico? (y/n): ").lower() == 'y':
                dm.create_config_from_detection()
        
    elif choice == "2":
        # Carica configurazione
        config_file = input("Nome file configurazione (default: data_config.json): ").strip()
        if not config_file:
            config_file = "data_config.json"
        dm.load_config(config_file)
        
    elif choice == "3":
        # Configurazione interattiva
        dm.interactive_config_creation()
        
    elif choice == "4":
        # Configurazione di esempio
        dm.create_sample_config()
        
    elif choice == "5":
        # Validazione
        if dm.config:
            dm.validate_files()
        else:
            print("❌ Carica prima una configurazione")
    
    # Se abbiamo una configurazione, mostra come usarla
    if dm.config:
        print(f"\n💡 Per usare questa configurazione nel training:")
        print(f"   file_list = dm.get_file_list_for_training()")
        print(f"   estimator.prepare_dataset(file_list)")


if __name__ == "__main__":
    main()
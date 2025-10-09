#!/usr/bin/env python3
"""
Script di test rapido per il sistema di stima diametro
Autore: Assistente AI
Data: 2025

Script semplificato per testare rapidamente il modello addestrato
"""

import os
import sys
from pathlib import Path
import argparse
from test_script import TimeSeriesDiameterTester

def check_requirements():
    """Verifica che i requirements siano installati"""
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import scipy
        import joblib
        return True
    except ImportError as e:
        print(f"❌ Librerie mancanti: {e}")
        print("💡 Installa con: pip install -r requirements.txt")
        return False

def check_model_exists(model_path='timeseries_diameter_estimator.pkl'):
    """Verifica che il modello esista"""
    if os.path.exists(model_path):
        print(f"✅ Modello trovato: {model_path}")
        return True
    else:
        print(f"❌ Modello non trovato: {model_path}")
        print("💡 Esegui prima: python train_diameter_estimator.py")
        return False

def find_test_files():
    """Trova automaticamente file CSV per il test"""
    test_files = []
    
    # Cerca nella directory corrente
    for pattern in ['*.csv', 'data/*.csv', 'data/*/*.csv']:
        import glob
        files = glob.glob(pattern, recursive=True)
        test_files.extend(files)
    
    # Rimuovi duplicati
    test_files = list(set(test_files))
    
    print(f"🔍 Trovati {len(test_files)} file CSV:")
    for f in test_files[:10]:  # Mostra solo i primi 10
        print(f"   - {f}")
    if len(test_files) > 10:
        print(f"   ... e altri {len(test_files) - 10} file")
    
    return test_files

def quick_test_single_file(file_path):
    """Test rapido su singolo file"""
    print(f"\n🧪 TEST RAPIDO SU: {file_path}")
    print("-" * 40)
    
    try:      
        # Inizializza tester
        tester = TimeSeriesDiameterTester()
        
        # Cerca di indovinare il diametro dal nome file
        filename = Path(file_path).name.lower()
        expected_diameter = None
        bar_type = 'unknown'
        
        # Pattern comuni per diametro
        import re
        diameter_match = re.search(r'd?(\d+)', filename)
        if diameter_match:
            expected_diameter = int(diameter_match.group(1))
        
        # Pattern per tipo
        if any(word in filename for word in ['liscio', 'lisce', 'smooth']):
            bar_type = 'liscio'
        elif any(word in filename for word in ['nervato', 'creste', 'ribbed']):
            bar_type = 'nervato'
        
        print(f"📊 Parametri rilevati: D{expected_diameter}mm, tipo: {bar_type}")
        
        # Esegui test
        result = tester.test_single_file(
            file_path, 
            expected_diameter, 
            bar_type, 
            plot=False  # No plot per test rapido
        )
        
        if result:
            print(f"✅ TEST RIUSCITO!")
            print(f"🎯 Diametro stimato: {result['mean_prediction']:.3f} mm")
            if result['expected_diameter']:
                print(f"🎲 Diametro atteso: {result['expected_diameter']} mm")
                print(f"❌ Errore: {result['relative_error']:.2f}%")
            print(f"📏 Stabilità: {result['cv_percent']:.2f}% CV")
            return True
        else:
            print(f"❌ TEST FALLITO")
            return False
            
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        return False

def interactive_mode():
    """Modalità interattiva semplificata"""
    print("\n🎛️  MODALITÀ INTERATTIVA RAPIDA")
    print("=" * 40)
    
    # Trova file automaticamente
    test_files = find_test_files()
    
    if not test_files:
        print("❌ Nessun file CSV trovato")
        file_path = input("📁 Inserisci percorso file CSV: ").strip()
        if file_path and os.path.exists(file_path):
            quick_test_single_file(file_path)
        else:
            print("❌ File non valido")
        return
    
    while True:
        print(f"\n📋 FILE DISPONIBILI:")
        for i, f in enumerate(test_files[:10], 1):
            print(f"  {i}) {f}")
        
        if len(test_files) > 10:
            print(f"  ... e altri {len(test_files) - 10} file")
        
        print(f"  0) Inserisci percorso manuale")
        print(f"  q) Esci")
        
        choice = input("\n🎯 Scegli file da testare: ").strip()
        
        if choice.lower() == 'q':
            break
        elif choice == '0':
            file_path = input("📁 Percorso file: ").strip()
            if file_path and os.path.exists(file_path):
                quick_test_single_file(file_path)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(test_files):
                    quick_test_single_file(test_files[idx])
                else:
                    print("❌ Scelta non valida")
            except ValueError:
                print("❌ Inserisci un numero valido")

def batch_test_mode():
    """Test batch su tutti i file trovati"""
    print("\n🧪 TEST BATCH AUTOMATICO")
    print("=" * 30)
    
    test_files = find_test_files()
    
    if not test_files:
        print("❌ Nessun file trovato per il test batch")
        return
    
    if len(test_files) > 20:
        print(f"⚠️  Trovati {len(test_files)} file - potrebbe richiedere tempo")
        if input("Continuare? (y/n): ").lower() != 'y':
            return
    
    try:
        
        tester = TimeSeriesDiameterTester()
        
        # Prepara lista file
        file_list = []
        for file_path in test_files:
            # Prova a rilevare parametri dal nome
            filename = Path(file_path).name.lower()
            
            import re
            diameter_match = re.search(r'd?(\d+)', filename)
            expected_diameter = int(diameter_match.group(1)) if diameter_match else None
            
            bar_type = 'unknown'
            if any(word in filename for word in ['liscio', 'lisce', 'smooth']):
                bar_type = 'liscio'
            elif any(word in filename for word in ['nervato', 'creste', 'ribbed']):
                bar_type = 'nervato'
            
            file_list.append((file_path, expected_diameter, bar_type))
        
        # Esegui test batch
        results = tester.batch_test(file_list)
        
        print(f"\n🎉 Test batch completato su {len(results)} file!")
        
    except Exception as e:
        print(f"❌ Errore test batch: {e}")

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Test rapido sistema tondini')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Modalità interattiva')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Test batch automatico')
    parser.add_argument('--file', '-f', help='Test singolo file')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Verifica setup')
    
    args = parser.parse_args()
    
    print("🚀 QUICK TEST - SISTEMA STIMA DIAMETRO TONDINI")
    print("=" * 50)
    
    # Verifica requirements
    if not check_requirements():
        return
    
    # Verifica modello
    if not check_model_exists():
        if args.check:
            print("\n💡 COME PROCEDERE:")
            print("1. pip install -r requirements.txt")
            print("2. python train_diameter_estimator.py")
            print("3. python quick_test.py --interactive")
        return
    
    print("✅ Setup verificato correttamente!")
    
    if args.check:
        print("\n🎉 Sistema pronto per l'uso!")
        return
    
    # Esegui il tipo di test richiesto
    if args.file:
        if os.path.exists(args.file):
            quick_test_single_file(args.file)
        else:
            print(f"❌ File non trovato: {args.file}")
    
    elif args.batch:
        batch_test_mode()
    
    elif args.interactive:
        interactive_mode()
    
    else:
        # Default: menu interattivo
        print("\n📖 OPZIONI DISPONIBILI:")
        print("1) Test interattivo")
        print("2) Test batch automatico") 
        print("3) Verifica setup")
        print("4) Esci")
        
        choice = input("\n🎯 Scelta: ").strip()
        
        if choice == '1':
            interactive_mode()
        elif choice == '2':
            batch_test_mode()
        elif choice == '3':
            print("✅ Setup già verificato!")
        elif choice == '4':
            print("👋 Ciao!")
        else:
            print("❌ Scelta non valida")

if __name__ == "__main__":
    main()
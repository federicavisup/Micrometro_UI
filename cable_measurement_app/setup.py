#!/usr/bin/env python3
"""
Script di setup per l'applicazione di misurazione cavi
"""

import subprocess
import sys
import os

def install_dependencies():
    """Installa le dipendenze necessarie"""
    print("=" * 60)
    print("Installazione dipendenze per Cable Measurement App")
    print("=" * 60)
    print()
    
    # Verifica versione Python
    if sys.version_info < (3, 8):
        print("❌ ERRORE: Richiesto Python 3.8 o superiore")
        print(f"   Versione corrente: {sys.version}")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} rilevato")
    print()
    
    # Lista delle dipendenze
    dependencies = [
        "PyQt6==6.6.1",
        "pyqtgraph==0.13.3",
        "numpy==1.26.2"
    ]
    
    print("Installazione delle seguenti dipendenze:")
    for dep in dependencies:
        print(f"  - {dep}")
    print()
    
    try:
        # Aggiorna pip
        print("Aggiornamento pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ pip aggiornato")
        print()
        
        # Installa dipendenze
        print("Installazione dipendenze...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements.txt"
        ])
        print()
        print("=" * 60)
        print("✓ Installazione completata con successo!")
        print("=" * 60)
        print()
        print("Per avviare l'applicazione, esegui:")
        print("  python cable_measurement_app.py")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ ERRORE durante l'installazione")
        print("=" * 60)
        print(f"Dettagli: {e}")
        print()
        print("Prova ad installare manualmente con:")
        print("  pip install PyQt6 pyqtgraph numpy")
        sys.exit(1)

def check_installation():
    """Verifica che tutti i moduli siano installati correttamente"""
    print()
    print("Verifica installazione moduli...")
    print()
    
    modules = {
        "PyQt6": "PyQt6",
        "pyqtgraph": "pyqtgraph",
        "numpy": "numpy"
    }
    
    all_ok = True
    for display_name, module_name in modules.items():
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - NON INSTALLATO")
            all_ok = False
    
    print()
    if all_ok:
        print("✓ Tutti i moduli sono installati correttamente!")
    else:
        print("❌ Alcuni moduli non sono installati. Esegui di nuovo lo script di setup.")
    
    return all_ok

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_installation()
    else:
        install_dependencies()
        check_installation()

#!/usr/bin/env python3
"""
Script per testare che il setup funzioni correttamente
"""

import pandas as pd
import numpy as np
import sys
import os

def test_imports():
    """Testa che tutte le librerie necessarie siano installate"""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        import sklearn
        print("✅ scikit-learn")
        
        import scipy
        print("✅ scipy")
        
        import joblib
        print("✅ joblib")
        
        print("✅ Tutte le librerie importate correttamente!\n")
        return True
        
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        return False

def test_data_files():
    """Verifica che i file di dati esistano"""
    print("📁 Testing data files...")
    
    files = ['D8_liscio_1.csv', 'D8_creste_1.csv', 'D6_creste_1.csv']
    
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file} - trovato")
            
            # Test rapido di lettura
            try:
                df = pd.read_csv(file)
                print(f"   📊 {len(df)} righe, colonne: {list(df.columns)}")
                
                # Verifica formato tempo
                if 'Tempo' in df.columns:
                    sample_time = df['Tempo'].iloc[0]
                    print(f"   ⏰ Formato tempo: {sample_time}")
                
            except Exception as e:
                print(f"   ❌ Errore lettura: {e}")
        else:
            print(f"❌ {file} - NON trovato")
    
    print()

def test_time_parsing():
    """Testa il parsing delle stringhe tempo"""
    print("⏰ Testing time parsing...")
    
    # Simula la funzione di parsing
    def parse_time_test(time_str):
        try:
            if ':' in str(time_str):
                time_parts = str(time_str).split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                if len(seconds_parts) > 1:
                    milliseconds_str = seconds_parts[1].ljust(3, '0')[:3]
                    milliseconds = int(milliseconds_str)
                else:
                    milliseconds = 0
                
                total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                return total_seconds
            else:
                return float(time_str)
        except Exception as e:
            print(f"Errore parsing tempo '{time_str}': {e}")
            return None
    
    # Test cases
    test_times = ["15:59:59.080", "16:08:51.519", "16:22:12.993", "10:00:00"]
    
    for time_str in test_times:
        result = parse_time_test(time_str)
        print(f"   '{time_str}' -> {result} secondi")
    
    print("✅ Time parsing test completato\n")

def test_feature_extraction():
    """Testa l'estrazione delle features su dati simulati"""
    print("🔧 Testing feature extraction...")
    
    # Crea dati simulati
    np.random.seed(42)
    n_points = 100
    
    # Simula un tondino D8 con leggere variazioni
    dx_base = 8.0
    dy_base = 8.0
    
    # Aggiungi rumore e pattern periodici (per simulare nervature)
    time_vals = np.linspace(0, 1, n_points)  # 1 secondo
    dx_vals = dx_base + 0.1 * np.sin(2 * np.pi * 5 * time_vals) + 0.02 * np.random.randn(n_points)
    dy_vals = dy_base + 0.1 * np.cos(2 * np.pi * 5 * time_vals) + 0.02 * np.random.randn(n_points)
    
    print(f"   📊 Dati simulati: {n_points} punti")
    print(f"   📈 Dx range: [{dx_vals.min():.3f}, {dx_vals.max():.3f}]")
    print(f"   📈 Dy range: [{dy_vals.min():.3f}, {dy_vals.max():.3f}]")
    
    # Test calcolo features base
    try:
        features = {}
        features['dx_mean'] = np.mean(dx_vals)
        features['dy_mean'] = np.mean(dy_vals)
        features['dx_std'] = np.std(dx_vals)
        features['dy_std'] = np.std(dy_vals)
        features['geometric_mean'] = np.sqrt(features['dx_mean'] * features['dy_mean'])
        
        print(f"   🎯 Features estratte: {len(features)}")
        print(f"   🎯 Diametro equivalente: {features['geometric_mean']:.3f} mm")
        print("✅ Feature extraction test superato\n")
        
    except Exception as e:
        print(f"❌ Errore feature extraction: {e}\n")

def main():
    """Esegue tutti i test"""
    print("=" * 50)
    print("🚀 TEST SETUP SISTEMA TONDINI")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("❌ Test imports fallito - controlla requirements.txt")
        return False
    
    # Test file dati
    test_data_files()
    
    # Test parsing tempo
    test_time_parsing()
    
    # Test estrazione features
    test_feature_extraction()
    
    print("=" * 50)
    print("🎉 TUTTI I TEST COMPLETATI")
    print("=" * 50)
    print("💡 Se tutto è OK, puoi procedere con:")
    print("   python train_diameter_estimator.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    main()
"""
Script di inferenza per predire il diametro equivalente di un cavo
a partire da un file di misure CSV.

Uso da linea di comando:
    python predict.py --file D30_1.csv
    python predict.py --file D30_1.csv --data_folder data --model_dir models
    python predict.py --file D30_1.csv --quiet
    
Uso programmatico:
    from predict import predict_diameter
    result = predict_diameter('D30_1.csv')
    print(f"Diametro stimato: {result['estimated_diameter']:.3f} mm")
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import common as utils


class CableDiameterPredictor:
    """
    Classe per caricare un modello addestrato e fare predizioni
    su nuovi file di misure.
    """
    
    def __init__(self, model_path, scaler_path, metadata_path):
        """
        Inizializza il predictor caricando modello, scaler e metadata.
        
        Args:
            model_path: path al file del modello (.joblib)
            scaler_path: path al file dello scaler (.joblib)
            metadata_path: path al file dei metadata (.json)
        """
        print("=" * 60)
        print("CARICAMENTO MODELLO")
        print("=" * 60)
        
        # Carica il modello
        self.model = joblib.load(model_path)
        print(f"✓ Modello caricato: {model_path}")
        
        # Carica lo scaler
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler caricato: {scaler_path}")
        
        # Carica i metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"✓ Metadata caricati: {metadata_path}")
        
        # Estrai parametri dai metadata
        self.model_name = self.metadata['model_name']
        self.window_size = self.metadata.get('window_size', 1.0)
        self.window_overlap = self.metadata.get('window_overlap', 0.5)
        self.selected_features = self.metadata.get('selected_features', None)
        
        print(f"\nConfigurazione modello:")
        print(f"  Nome: {self.model_name}")
        print(f"  Tipo: {self.metadata['model_type']}")
        print(f"  Window size: {self.window_size}s")
        print(f"  Window overlap: {self.window_overlap*100:.0f}%")
        print(f"  Features selezionate: {len(self.selected_features) if self.selected_features else 'tutte'}")
        
        print(f"\nMetriche sul test set durante il training:")
        for metric, value in self.metadata['test_metrics'].items():
            if value != 'inf':
                if metric == 'MAPE':
                    print(f"  {metric}: {value:.2f}%")
                else:
                    print(f"  {metric}: {value:.4f}")
        
        print("=" * 60 + "\n")
    
    def predict_from_file(self, filepath, data_folder='data', verbose=True):
        """
        Predice il diametro equivalente da un file CSV.
        
        Args:
            filepath: nome del file CSV (es: 'D30_1.csv') o path completo
            data_folder: cartella dove cercare il file (se filepath è solo il nome)
            verbose: se True, stampa informazioni dettagliate
            
        Returns:
            dict con risultati della predizione
        """
        if verbose:
            print("=" * 60)
            print("PREDIZIONE DIAMETRO EQUIVALENTE")
            print("=" * 60)
        
        # Costruisci il path completo se necessario
        if not os.path.isabs(filepath):
            filepath = Path(data_folder) / filepath
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        if verbose:
            print(f"File: {filepath.name}")
        
        # 1. Carica e valida i dati
        if verbose:
            print("\n1. Caricamento e validazione dati...")
        
        df = pd.read_csv(filepath)
        df_clean, removed = utils.validate_and_clean_data(df, filepath.name)
        
        if verbose:
            print(f"   ✓ Misure valide: {len(df_clean)}")
            if removed > 0:
                print(f"   ⚠ Misure rimosse: {removed}")
        
        # 2. Crea sliding windows ed estrai features
        if verbose:
            print(f"\n2. Estrazione features (finestre da {self.window_size}s)...")
        
        dx_series = df_clean['Dx'].values
        dy_series = df_clean['Dy'].values
        
        windows_features = utils.create_sliding_windows(
            dx_series, dy_series,
            window_size_seconds=self.window_size,
            window_overlap=self.window_overlap,
            estimated_duration=10.0
        )
        
        if len(windows_features) == 0:
            raise ValueError("Nessuna finestra valida estratta dal file!")
        
        if verbose:
            print(f"   ✓ Finestre create: {len(windows_features)}")
        
        # 3. Converti in DataFrame
        features_df = pd.DataFrame(windows_features)
        
        # 4. Seleziona le features usate nel training
        if self.selected_features is not None:
            # Usa solo le features selezionate durante il training
            try:
                X = features_df[self.selected_features]
            except KeyError as e:
                missing = set(self.selected_features) - set(features_df.columns)
                raise ValueError(f"Features mancanti: {missing}. Verifica la compatibilità del modello.")
        else:
            # Usa tutte le features
            X = features_df
        
        if verbose:
            print(f"   ✓ Features utilizzate: {X.shape[1]}")
        
        # 5. Scala le features
        X_scaled = self.scaler.transform(X)
        
        # 6. Predizione
        if verbose:
            print("\n3. Predizione in corso...")
        
        predictions = self.model.predict(X_scaled)
        
        # 7. Calcola statistiche
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        min_prediction = np.min(predictions)
        max_prediction = np.max(predictions)
        median_prediction = np.median(predictions)
        
        # 8. Risultati
        if verbose:
            print("\n" + "=" * 60)
            print("RISULTATI")
            print("=" * 60)
            print(f"\n📊 Statistiche predizioni ({len(predictions)} finestre):")
            print(f"   Media:    {mean_prediction:.3f} mm")
            print(f"   Mediana:  {median_prediction:.3f} mm")
            print(f"   Std Dev:  {std_prediction:.3f} mm")
            print(f"   Min:      {min_prediction:.3f} mm")
            print(f"   Max:      {max_prediction:.3f} mm")
            print(f"\n🎯 DIAMETRO EQUIVALENTE STIMATO: {mean_prediction/10:.3f} mm")
            print("=" * 60 + "\n")
        
        # Prepara il dizionario dei risultati
        result = {
            'filename': filepath.name,
            'estimated_diameter': mean_prediction,
            'median_diameter': median_prediction,
            'std_diameter': std_prediction,
            'min_diameter': min_prediction,
            'max_diameter': max_prediction,
            'n_windows': len(predictions),
            'all_predictions': predictions.tolist(),
            'model_used': self.model_name
        }
        
        return result


def find_latest_model(model_dir='models'):
    """
    Trova i file del modello più recente nella cartella models.
    
    Args:
        model_dir: cartella dove cercare i modelli
        
    Returns:
        tuple (model_path, scaler_path, metadata_path)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Cartella modelli non trovata: {model_dir}")
    
    # Cerca i file più recenti
    model_files = sorted(model_path.glob("best_model_*.joblib"), reverse=True)
    scaler_files = sorted(model_path.glob("scaler_*.joblib"), reverse=True)
    metadata_files = sorted(model_path.glob("metadata_*.json"), reverse=True)
    
    if not model_files or not scaler_files or not metadata_files:
        raise FileNotFoundError(f"File del modello non trovati in {model_dir}")
    
    # Prendi i più recenti (stesso timestamp idealmente)
    model_file = model_files[0]
    scaler_file = scaler_files[0]
    metadata_file = metadata_files[0]
    
    return str(model_file), str(scaler_file), str(metadata_file)


def predict_diameter(filename, data_folder='data', model_dir='models', 
                     model_path=None, scaler_path=None, metadata_path=None,
                     verbose=True):
    """
    Funzione di utilità per predire il diametro equivalente da un file.
    
    Args:
        filename: nome del file CSV da predire
        data_folder: cartella dove si trova il file
        model_dir: cartella dove si trovano i modelli (se model_path non specificato)
        model_path: path specifico al modello (opzionale)
        scaler_path: path specifico allo scaler (opzionale)
        metadata_path: path specifico ai metadata (opzionale)
        verbose: se True, stampa informazioni
        
    Returns:
        dict con i risultati della predizione
    """
    # Se non specificati, trova i file del modello più recente
    if model_path is None or scaler_path is None or metadata_path is None:
        model_path, scaler_path, metadata_path = find_latest_model(model_dir)
    
    # Crea il predictor
    predictor = CableDiameterPredictor(model_path, scaler_path, metadata_path)
    
    # Esegui la predizione
    result = predictor.predict_from_file(filename, data_folder, verbose)
    
    return result


def main():
    """
    Funzione main per eseguire lo script da linea di comando.
    """
    parser = argparse.ArgumentParser(
        description='Predice il diametro equivalente di un cavo da un file CSV'
    )
    parser.add_argument('--file', '-f', required=False,
                       help='Nome del file CSV da analizzare (es: D30_1.csv). Se non specificato, verrà richiesto interattivamente.')
    parser.add_argument('--data_folder', '-d', default='data',
                       help='Cartella contenente i file CSV (default: data)')
    parser.add_argument('--model_dir', '-m', default='models',
                       help='Cartella contenente i modelli (default: models)')
    parser.add_argument('--model', help='Path specifico al file del modello')
    parser.add_argument('--scaler', help='Path specifico al file dello scaler')
    parser.add_argument('--metadata', help='Path specifico al file dei metadata')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modalità silenziosa (solo risultato finale)')
    
    args = parser.parse_args()
    
    # Se il file non è specificato come argomento, chiedi all'utente
    if args.file is None:
        print("=" * 60)
        print("PREDIZIONE DIAMETRO EQUIVALENTE - Input Interattivo")
        print("=" * 60)
        filename = input("\nInserisci il nome del file CSV (es: D30_1.csv): ").strip()
        
        if not filename:
            print("❌ ERRORE: Nome file non può essere vuoto!")
            return 1
    else:
        filename = args.file
    
    try:
        result = predict_diameter(
            filename=filename,
            data_folder=args.data_folder,
            model_dir=args.model_dir,
            model_path=args.model,
            scaler_path=args.scaler,
            metadata_path=args.metadata,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(f"{result['estimated_diameter']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERRORE: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
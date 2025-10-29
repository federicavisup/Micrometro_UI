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
import re
import math


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
        # Carica il modello silenziosamente
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Estrai parametri dai metadata
        self.model_name = self.metadata['model_name']
        self.window_size = self.metadata.get('window_size', 1.0)
        self.window_overlap = self.metadata.get('window_overlap', 0.5)
        self.selected_features = self.metadata.get('selected_features', None)
    
    def predict_from_file(self, filepath, data_folder='data', verbose=False):
        """
        Predice il diametro equivalente da un file CSV.
        
        Args:
            filepath: nome del file CSV (es: 'D30_1.csv') o path completo
            data_folder: cartella dove cercare il file (se filepath è solo il nome)
            verbose: se True, stampa informazioni dettagliate
            
        Returns:
            dict con risultati della predizione
        """
        # Costruisci il path completo se necessario
        if not os.path.isabs(filepath):
            filepath = Path(data_folder) / filepath
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File non trovato: {filepath}")
        
        # 1. Carica e valida i dati
        df = pd.read_csv(filepath)
        df_clean, removed = utils.validate_and_clean_data(df, filepath.name)
        
        # 2. Crea sliding windows ed estrai features
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
        
        # 5. Scala le features
        X_scaled = self.scaler.transform(X)
        
        # 6. Predizione
        predictions = self.model.predict(X_scaled)
        
        # 7. Calcola statistiche e ritorna senza prints
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        min_prediction = np.min(predictions)
        max_prediction = np.max(predictions)
        median_prediction = np.median(predictions)
        
        return {
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
                     verbose=False):
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
    parser.add_argument('--all', action='store_true',
                       help='Esegui predizione su tutti i file .csv nella cartella "test" dentro data_folder')
    parser.add_argument('--test_folder', default='test',
                       help='Nome della sottocartella con i file di test (default: test)')
    
    args = parser.parse_args()
    
    # Batch mode: processa tutti i .csv nella cartella data_folder/test_folder
    if args.all:
    #if True:
        test_dir = Path(args.data_folder) / args.test_folder
        if not test_dir.exists():
            print(f"❌ ERRORE: Cartella test non trovata: {test_dir}")
            return 1
        
        csv_files = sorted(test_dir.glob("*.csv"))
        if not csv_files:
            print(f"❌ ERRORE: Nessun file .csv trovato in {test_dir}")
            return 1
        
        true_vals = []
        pred_vals = []
        abs_errors = []
        sq_errors = []
        pct_errors = []
        processed = 0
        failed = 0
        
        # Rimuovi header batch
        for fp in csv_files:
            try:
                fname = fp.name
                # Estrai valore vero dal nome: cerca 'D<number>'
                m = re.search(r"D(\d+)", fname, re.IGNORECASE)
                if not m:
                    print(f"⚠ Skipping {fname}: pattern D<number> non trovato nel nome.")
                    failed += 1
                    continue
                true_value = float(m.group(1))/10.0
                
                # Esegui predizione (usa data_folder come path base)
                # passiamo il filename relativo alla funzione predict_diameter
                result = predict_diameter(
                    filename=str(fp),
                    data_folder="",
                    model_dir=args.model_dir,
                    model_path=args.model,
                    scaler_path=args.scaler,
                    metadata_path=args.metadata,
                    verbose=False  # sempre False per ridurre output
                )
                
                estimated_raw = result['estimated_diameter']  # valore come usato internamente
                predicted_mm = estimated_raw/10.0
                true_mm = float(true_value)
                
                abs_err = abs(predicted_mm - true_mm)
                sq_err = (predicted_mm - true_mm) ** 2
                pct_err = abs_err / true_mm * 100.0 if true_mm != 0 else float('inf')
                
                true_vals.append(true_mm)
                pred_vals.append(predicted_mm)
                abs_errors.append(abs_err)
                sq_errors.append(sq_err)
                if math.isfinite(pct_err):
                    pct_errors.append(pct_err)
                
                processed += 1
                
                # Stampa solo la riga essenziale per ogni file
                print(f"{fname}: true={true_mm:.3f} mm, pred={predicted_mm:.3f} mm, abs_err={abs_err:.3f} mm, pct_err={pct_err:.2f}%")
            
            except Exception as e:
                print(f"❌ Errore su {fp.name}: {e}")
                failed += 1
                continue
        
        # Metriche aggregate (mantieni questo output)
        if processed > 0:
            mae = float(np.mean(abs_errors))
            rmse = float(math.sqrt(np.mean(sq_errors)))
            mape = float(np.mean(pct_errors)) if pct_errors else float('inf')
            
            print("\n" + "=" * 60)
            print("RIASSUNTO METRICHE")
            print("=" * 60)
            print(f"  Files processati: {processed}")
            print(f"  Files falliti:    {failed}")
            print(f"  MAE:              {mae:.3f} mm")
            print(f"  RMSE:             {rmse:.3f} mm")
            if math.isfinite(mape):
                print(f"  MAPE:             {mape:.2f}%")
            else:
                print(f"  MAPE:             inf (divisione per zero)")
            # Brevi definizioni delle metriche
            print("\n  Note sulle metriche:")
            print("    MAE  = (errore assoluto medio in mm).")
            print("    RMSE = (radice dell'errore quadratico medio in mm; ).")
            print("    MAPE = (errore percentuale medio).")
            print("=" * 60)
        
        return 0
    
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
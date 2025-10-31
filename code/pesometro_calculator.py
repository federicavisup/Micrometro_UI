import os
import argparse
import math
from pathlib import Path
import pandas as pd
import sys

# Assicura che la cartella corrente (script) sia nel path per importare predict.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from predict import predict_diameter

def compute_weight_per_m(d_mm, rho=7850.0):
    # d_mm: diametro in millimetri
    d_m = d_mm / 1000.0
    area = math.pi * (d_m / 2.0) ** 2
    return rho * area

def process_single(file_path, args):
    reslts = []
    # chiama predict_diameter con il path completo
    result = predict_diameter(
        filename=str(file_path),
        data_folder="", 
        model_dir=args.model_dir,
        model_path=args.model,
        scaler_path=args.scaler,
        metadata_path=args.metadata,
        verbose=False
    )
    estimated_raw = result['all_predictions']
    # conversione coerente con il predict.py presente: /10
    for pd_mm in estimated_raw:
        predicted_mm = pd_mm / 10.0
        weight = compute_weight_per_m(predicted_mm)
        reslts.append({
            'filename': Path(file_path).name,
            'estimated_diameter_mm': pd_mm,
            'weight_kg_per_m': weight
        })
    return reslts

def main():
    parser = argparse.ArgumentParser(description='Calcola peso/m a partire da predizioni diametro')
    parser.add_argument('--file', '-f', help='Singolo file CSV (path o nome)')
    parser.add_argument('--data_folder', '-d', default='data', help='Cartella contenente i file CSV (default: data)')
    parser.add_argument('--test_folder', default='test', help='Sottocartella con file di test (default: test)')
    parser.add_argument('--all', action='store_true', help='Processa tutti i .csv nella cartella data_folder/test_folder')
    parser.add_argument('--model_dir', '-m', default='models', help='Cartella modelli (passata a predict)')
    parser.add_argument('--model', help='Path specifico al file del modello (passato a predict)')
    parser.add_argument('--scaler', help='Path specifico al file dello scaler (passato a predict)')
    parser.add_argument('--metadata', help='Path specifico al file dei metadata (passato a predict)')
    parser.add_argument('--output', '-o', help='Percorso output CSV (se non specificato: results.csv nella cartella dei file)')
    args = parser.parse_args()

    rows = []
    if True:
        test_dir = Path(args.data_folder) / args.test_folder
        if not test_dir.exists():
            print(f"❌ ERRORE: cartella test non trovata: {test_dir}")
            return 1
        csv_files = sorted(test_dir.glob("*.csv"))
        if not csv_files:
            print(f"❌ ERRORE: nessun .csv in {test_dir}")
            return 1
        for fp in csv_files:
            try:
                rowp = process_single(fp, args)
                for row in rowp:
                    rows.append(row)
                    print(f"{row['filename']}: true=.. mm, pred={row['estimated_diameter_mm']:.3f} mm, weight={row['weight_kg_per_m']:.6f} kg/m")
            except Exception as e:
                print(f"❌ Errore su {fp.name}: {e}")
                continue
        # salva results.csv in test_dir o in --output se fornito
        out_path = Path(args.output) if args.output else ( "results.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Risultati salvati in: {out_path}")
        return 0

    # singolo file
    if not args.file:
        print("Inserisci il nome/path del file con --file")
        return 1
    file_path = Path(args.file)
    if not file_path.is_absolute():
        # primo cerca in data_folder, poi come percorso relativo allo script
        candidate = Path(args.data_folder) / file_path
        if candidate.exists():
            file_path = candidate
        else:
            file_path = file_path if file_path.exists() else None
    if file_path is None or not Path(file_path).exists():
        print(f"❌ ERRORE: file non trovato: {args.file}")
        return 1
    try:
        row = process_single(file_path, args)
        rows.append(row)
        print(f"{row['filename']}: pred={row['estimated_diameter_mm']:.3f} mm, weight={row['weight_kg_per_m']:.6f} kg/m")
        out_path = Path(args.output) if args.output else (Path(file_path).parent / "results.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Risultati salvati in: {out_path}")
        return 0
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

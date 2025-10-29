import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats

def extract_nominal_diameter(filename):
    """Estrae il diametro nominale dal nome file (es: D117_1.csv -> 117)"""
    m = re.search(r"D(\d+)", filename)
    return int(m.group(1)) if m else None

def calculate_statistics(group):
    """Calcola statistiche per un gruppo di misure dello stesso diametro nominale"""
    try:
        if len(group) == 0:
            return pd.Series({
                'count': 0, 'mean': np.nan, 
                'std': np.nan, 'min': np.nan,
                'max': np.nan, 'range': np.nan, 
                'cv_percent': np.nan
            })
        
        return pd.Series({
            'count': len(group),
            'mean': group.mean(),
            'std': group.std() if len(group) > 1 else 0,
            'min': group.min(),
            'max': group.max(),
            'range': group.max() - group.min(),
            'cv_percent': (group.std() / group.mean() * 100) if (len(group) > 1 and group.mean() != 0) else 0
        })
    except Exception as e:
        print(f"Errore nel calcolo statistiche: {e}")
        return pd.Series({
            'count': len(group), 'mean': np.nan,
            'std': np.nan, 'min': np.nan,
            'max': np.nan, 'range': np.nan,
            'cv_percent': np.nan
        })

def analyze_results(results_path):
    """Analizza i risultati dal CSV e calcola statistiche (ora su peso/m)"""
    try:
        # Leggi results.csv e validazione base
        df = pd.read_csv(results_path)
        if df.empty:
            raise ValueError("Il file results.csv è vuoto")
        
        # Verifica colonne necessarie
        required_cols = {'filename', 'estimated_diameter_mm', 'weight_kg_per_m'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Colonne necessarie mancanti nel CSV. Richieste: {required_cols}")
        
        # Estrai diametro nominale e calcola peso vero
        df['nominal_diameter'] = df['filename'].apply(extract_nominal_diameter)
        if df['nominal_diameter'].isna().all():
            raise ValueError("Impossibile estrarre i diametri nominali dai nomi file")
        
        rho = 7850.0  # kg/m^3
        error_rows = []
        total_errors = []
        total_count = 0
        
        # Calcola statistiche per ogni diametro nominale
        for nominal, grp in df.groupby('nominal_diameter'):
            true_mm = nominal / 10.0
            d_m = true_mm / 1000.0
            true_weight = rho * (np.pi * (d_m / 2.0) ** 2)
            
            pred_weights = grp['weight_kg_per_m'].values
            errors = pred_weights - true_weight
            abs_errors = np.abs(errors)
            count = len(pred_weights)
            total_count += count
            total_errors.extend(errors.tolist())
            
            mae = float(np.mean(abs_errors))
            rmse = float(np.sqrt(np.mean(errors**2)))
            mape = float(np.mean((abs_errors / true_weight) * 100.0))
            bias = float(np.mean(errors))
            err_std = float(np.std(errors))  # Standard deviation of errors
            
            # Percentuali di misure entro certi limiti
            pct_within_0_001 = np.mean(abs_errors <= 0.001) * 100
            pct_within_1pct = np.mean(abs_errors / true_weight <= 0.01) * 100
            
            error_rows.append({
                'nominal': nominal,
                'true_weight_kg_per_m': true_weight,
                'count': count,
                'MAE_kg_per_m': mae,
                'RMSE_kg_per_m': rmse,
                'MAPE_%': mape,
                'bias_kg_per_m': bias,
                'err_std_kg_per_m': err_std,
                'pct_within_0.001kg_m': pct_within_0_001,
                'pct_within_1%': pct_within_1pct
            })
        
        error_df = pd.DataFrame(error_rows).set_index('nominal').sort_index()
        
        # Statistiche globali complete
        if total_count > 0:
            total_errors = np.array(total_errors)
            overall_mae = float(np.mean(np.abs(total_errors)))
            overall_rmse = float(np.sqrt(np.mean(total_errors**2)))
            overall_mape = float(np.mean((np.abs(total_errors) / np.abs(true_weight)) * 100.0))
            overall_bias = float(np.mean(total_errors))
            overall_err_std = float(np.std(total_errors))
            
            return error_df, {
                'overall_MAE_kg_per_m': overall_mae,
                'overall_RMSE_kg_per_m': overall_rmse,
                'overall_MAPE_%': overall_mape,
                'overall_bias_kg_per_m': overall_bias,
                'overall_err_std_kg_per_m': overall_err_std,
                'total_count': total_count
            }
        
    except Exception as e:
        print(f"Errore nell'analisi dei risultati: {e}")
        raise

def main():
    """Analizza performance sul peso/m rispetto al valore vero"""
    results_path = Path("results.csv")
    if not results_path.exists():
        print(f"❌ ERRORE: File non trovato: {results_path}")
        return 1
    
    try:
        error_df, overall = analyze_results(results_path)
        
        # Output statistiche di errore sul peso per diametro
        print("\n" + "=" * 80)
        print("STATISTICHE DI ERRORE SUL PESO/M PER DIAMETRO NOMINALE")
        print("=" * 80)
        print(error_df.round(6))
        
        # Output metriche globali complete
        print("\n" + "=" * 80)
        print("METRICHE GLOBALI DI BONTÀ (sul peso per metro)")
        print("=" * 80)
        print(f"Numero totale misure: {int(overall['total_count'])}")
        print(f"MAE globale:  {overall['overall_MAE_kg_per_m']:.6f} kg/m")
        print(f"RMSE globale: {overall['overall_RMSE_kg_per_m']:.6f} kg/m")
        print(f"MAPE globale: {overall['overall_MAPE_%']:.3f} %")
        print(f"Bias (mean error): {overall['overall_bias_kg_per_m']:.6f} kg/m")
        print(f"Std degli errori:   {overall['overall_err_std_kg_per_m']:.6f} kg/m")
        
        print("\nNote:")
        print("- Le statistiche sono calcolate sul peso per metro (kg/m). Il valore vero è calcolato dal diametro nominale.")
        print("- MAE = media degli errori assoluti; RMSE = radice media degli errori quadrati")
        print("- MAPE = errore percentuale medio; Bias = errore sistematico (media errori)")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERRORE durante l'analisi: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

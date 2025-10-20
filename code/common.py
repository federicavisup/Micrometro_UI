"""
Modulo condiviso per funzioni di preprocessing e feature extraction
per il sistema ML di stima diametro equivalente cavi.

Questo modulo è condiviso tra train.py e predict.py per garantire
consistenza nelle operazioni di preprocessing e feature extraction.
"""

import pandas as pd
import numpy as np


def validate_and_clean_data(df, filename):
    """
    Valida e pulisce i dati da un DataFrame.
    Rimuove righe con valori nulli o negativi in Dx/Dy.
    
    Args:
        df: DataFrame con colonne Tempo, Dx, Dy
        filename: nome del file (per logging)
        
    Returns:
        DataFrame pulito, numero di righe rimosse
    """
    # Verifica che le colonne necessarie esistano
    required_cols = ['Tempo', 'Dx', 'Dy']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Colonne mancanti in {filename}. Richieste: {required_cols}")
    
    original_len = len(df)
    
    # Rimuovi righe con valori nulli
    df_clean = df.dropna()
    
    # Rimuovi righe con valori negativi in Dx o Dy
    df_clean = df_clean[(df_clean['Dx'] > 0) & (df_clean['Dy'] > 0)]
    
    # Verifica che rimangano dati validi
    if len(df_clean) < 5:
        raise ValueError(f"Troppo poche misure valide in {filename} ({len(df_clean)} misure)")
    
    removed = original_len - len(df_clean)
    
    return df_clean, removed


def extract_rib_features(dx_series, dy_series, window_size_seconds):
    """
    Estrae features relative alle nervature analizzando picchi e valli.
    
    Le nervature causano oscillazioni periodiche nel segnale. Questa funzione
    rileva i picchi (massimi locali) e le valli (minimi locali) e calcola
    le ampiezze picco-valle che rappresentano la dimensione delle nervature.
    
    Args:
        dx_series: array numpy di misure Dx
        dy_series: array numpy di misure Dy
        window_size_seconds: durata della finestra in secondi
        
    Returns:
        dict con features delle nervature
    """
    rib_features = {}
    
    if len(dx_series) > 10:
        # Funzione per trovare picchi e valli locali
        def find_peaks_valleys(signal, window_size=5):
            """
            Trova picchi (massimi locali) e valli (minimi locali) in un segnale.
            Un punto è un picco se è il massimo in una finestra locale.
            Un punto è una valle se è il minimo in una finestra locale.
            """
            peaks = []
            valleys = []
            
            for i in range(window_size, len(signal) - window_size):
                window = signal[i-window_size:i+window_size+1]
                if signal[i] == np.max(window):
                    peaks.append(signal[i])
                elif signal[i] == np.min(window):
                    valleys.append(signal[i])
            
            return np.array(peaks), np.array(valleys)
        
        # Trova picchi e valli per Dx e Dy
        dx_peaks, dx_valleys = find_peaks_valleys(dx_series)
        dy_peaks, dy_valleys = find_peaks_valleys(dy_series)
        
        # Calcola ampiezze picco-valle (nervature)
        dx_amplitudes = []
        dy_amplitudes = []
        
        if len(dx_peaks) > 0 and len(dx_valleys) > 0:
            dx_amplitudes = [abs(p - v) for p in dx_peaks for v in dx_valleys]
        
        if len(dy_peaks) > 0 and len(dy_valleys) > 0:
            dy_amplitudes = [abs(p - v) for p in dy_peaks for v in dy_valleys]
        
        # Features basate sull'ampiezza delle nervature
        if len(dx_amplitudes) > 0:
            rib_features['Dx_rib_amplitude_max'] = np.max(dx_amplitudes)
            rib_features['Dx_rib_amplitude_mean'] = np.mean(dx_amplitudes)
            rib_features['Dx_rib_amplitude_std'] = np.std(dx_amplitudes)
        else:
            rib_features['Dx_rib_amplitude_max'] = 0
            rib_features['Dx_rib_amplitude_mean'] = 0
            rib_features['Dx_rib_amplitude_std'] = 0
        
        if len(dy_amplitudes) > 0:
            rib_features['Dy_rib_amplitude_max'] = np.max(dy_amplitudes)
            rib_features['Dy_rib_amplitude_mean'] = np.mean(dy_amplitudes)
            rib_features['Dy_rib_amplitude_std'] = np.std(dy_amplitudes)
        else:
            rib_features['Dy_rib_amplitude_max'] = 0
            rib_features['Dy_rib_amplitude_mean'] = 0
            rib_features['Dy_rib_amplitude_std'] = 0
        
        # Feature combinata
        all_amplitudes = dx_amplitudes + dy_amplitudes
        if len(all_amplitudes) > 0:
            rib_features['combined_rib_amplitude'] = np.mean(all_amplitudes)
            rib_features['combined_rib_max'] = np.max(all_amplitudes)
        else:
            rib_features['combined_rib_amplitude'] = 0
            rib_features['combined_rib_max'] = 0
        
        # Frequenza picchi (normalizzata per la durata della finestra)
        rib_features['Dx_peaks_per_second'] = len(dx_peaks) / window_size_seconds if len(dx_peaks) > 0 else 0
        rib_features['Dy_peaks_per_second'] = len(dy_peaks) / window_size_seconds if len(dy_peaks) > 0 else 0
        
    else:
        rib_features = {
            'Dx_rib_amplitude_max': 0,
            'Dx_rib_amplitude_mean': 0,
            'Dx_rib_amplitude_std': 0,
            'Dy_rib_amplitude_max': 0,
            'Dy_rib_amplitude_mean': 0,
            'Dy_rib_amplitude_std': 0,
            'combined_rib_amplitude': 0,
            'combined_rib_max': 0,
            'Dx_peaks_per_second': 0,
            'Dy_peaks_per_second': 0,
        }
    
    return rib_features


def extract_features_from_window(dx_window, dy_window, window_size_seconds):
    """
    Estrae tutte le features da una finestra temporale di dati.
    
    Args:
        dx_window: array numpy di misure Dx per la finestra
        dy_window: array numpy di misure Dy per la finestra
        window_size_seconds: durata della finestra in secondi
        
    Returns:
        dict con tutte le features estratte
    """
    # Calcola features statistiche per Dx
    dx_features = {
        'Dx_mean': np.mean(dx_window),
        'Dx_std': np.std(dx_window),
        'Dx_min': np.min(dx_window),
        'Dx_max': np.max(dx_window),
        'Dx_median': np.median(dx_window),
        'Dx_q25': np.percentile(dx_window, 25),
        'Dx_q75': np.percentile(dx_window, 75),
        'Dx_range': np.max(dx_window) - np.min(dx_window),
    }
    
    # Calcola features statistiche per Dy
    dy_features = {
        'Dy_mean': np.mean(dy_window),
        'Dy_std': np.std(dy_window),
        'Dy_min': np.min(dy_window),
        'Dy_max': np.max(dy_window),
        'Dy_median': np.median(dy_window),
        'Dy_q25': np.percentile(dy_window, 25),
        'Dy_q75': np.percentile(dy_window, 75),
        'Dy_range': np.max(dy_window) - np.min(dy_window),
    }
    
    # Features combinate
    combined_features = {
        'mean_diff': np.mean(dx_window) - np.mean(dy_window),
        'mean_ratio': np.mean(dx_window) / np.mean(dy_window) if np.mean(dy_window) != 0 else 0,
        'std_diff': np.std(dx_window) - np.std(dy_window),
        'std_ratio': np.std(dx_window) / np.std(dy_window) if np.std(dy_window) != 0 else 0,
        'cv_dx': np.std(dx_window) / np.mean(dx_window) if np.mean(dx_window) != 0 else 0,
        'cv_dy': np.std(dy_window) / np.mean(dy_window) if np.mean(dy_window) != 0 else 0,
    }
    
    # Features nervature
    rib_features = extract_rib_features(dx_window, dy_window, window_size_seconds)
    
    # Combina tutte le features
    all_features = {**dx_features, **dy_features, **combined_features, **rib_features}
    
    return all_features


def create_sliding_windows(dx_series, dy_series, window_size_seconds=1.0, 
                          window_overlap=0.5, estimated_duration=10.0):
    """
    Crea sliding windows da serie temporali di dati e estrae features.
    
    Args:
        dx_series: array numpy di misure Dx
        dy_series: array numpy di misure Dy
        window_size_seconds: durata della finestra in secondi
        window_overlap: overlap tra finestre (0.5 = 50%)
        estimated_duration: durata stimata dell'acquisizione in secondi
        
    Returns:
        Lista di dict, ognuno contenente le features per una finestra
    """
    # Calcola il sampling rate medio
    total_samples = len(dx_series)
    samples_per_second = total_samples / estimated_duration
    
    # Calcola quanti campioni servono per una finestra
    window_samples = int(window_size_seconds * samples_per_second)
    
    # Calcola lo step per lo sliding window
    step_samples = int(window_samples * (1 - window_overlap))
    
    # Verifica che la finestra non sia troppo piccola
    if window_samples < 10:
        # Se troppo piccola, usa tutto
        window_samples = total_samples
        step_samples = window_samples
    
    # Crea sliding windows
    windows_features = []
    
    for start_idx in range(0, total_samples - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        
        # Estrai la finestra
        dx_window = dx_series[start_idx:end_idx]
        dy_window = dy_series[start_idx:end_idx]
        
        # Salta finestre troppo piccole
        if len(dx_window) < 10:
            continue
        
        # Estrai features dalla finestra
        features = extract_features_from_window(dx_window, dy_window, window_size_seconds)
        windows_features.append(features)
    
    return windows_features


def get_feature_names():
    """
    Restituisce i nomi di tutte le features nell'ordine corretto.
    Deve corrispondere all'ordine usato in extract_features_from_window.
    
    Returns:
        Lista di nomi delle features
    """
    dx_names = ['Dx_mean', 'Dx_std', 'Dx_min', 'Dx_max', 'Dx_median', 'Dx_q25', 'Dx_q75', 'Dx_range']
    dy_names = ['Dy_mean', 'Dy_std', 'Dy_min', 'Dy_max', 'Dy_median', 'Dy_q25', 'Dy_q75', 'Dy_range']
    combined_names = ['mean_diff', 'mean_ratio', 'std_diff', 'std_ratio', 'cv_dx', 'cv_dy']
    rib_names = [
        'Dx_rib_amplitude_max', 'Dx_rib_amplitude_mean', 'Dx_rib_amplitude_std',
        'Dy_rib_amplitude_max', 'Dy_rib_amplitude_mean', 'Dy_rib_amplitude_std',
        'combined_rib_amplitude', 'combined_rib_max',
        'Dx_peaks_per_second', 'Dy_peaks_per_second'
    ]
    
    return dx_names + dy_names + combined_names + rib_names
#!/usr/bin/env python3
"""
Script per il testing e validazione del modello di stima del diametro equivalente
basato su serie temporali
Autore: Assistente AI
Data: 2025

Questo script carica un modello pre-addestrato e lo testa su nuovi dati temporali,
fornendo predizioni del diametro equivalente e analisi delle prestazioni.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import argparse
from pathlib import Path
from data_manager import TondinoDataManager

warnings.filterwarnings('ignore')

class TimeSeriesDiameterTester:
    def __init__(self, model_path='timeseries_diameter_estimator.pkl'):
        """
        Inizializza il tester caricando il modello pre-addestrato
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        
        print(f"📂 Caricamento modello da {model_path}...")
        model_data = joblib.load(model_path)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.feature_importance = model_data.get('feature_importance', {})
        self.feature_names = model_data.get('feature_names', [])
        self.window_duration = model_data.get('window_duration', 1.0)
        self.overlap = model_data.get('overlap', 0.5)
        self.all_models = model_data.get('models', {})
        
        print(f"✅ Modello caricato: {self.best_model_name}")
        print(f"🔧 Configurazione: finestre {self.window_duration}s, overlap {self.overlap}")
        print(f"🎯 Features: {len(self.feature_names)}")
    
    def parse_time_column(self, time_str):
        """
        Converte la stringa tempo in secondi relativi (identico al training)
        """
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
    
    def load_and_clean_data(self, file_path, expected_diameter=None, bar_type='unknown'):
        """
        Carica e pulisce i dati da un file CSV (identico al training)
        """
        print(f"📁 Caricamento dati da {file_path}...")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Converti colonna tempo
        df['time_seconds'] = df['Tempo'].apply(self.parse_time_column)
        df = df.dropna(subset=['time_seconds'])
        
        # Normalizza il tempo
        df['time_seconds'] = df['time_seconds'] - df['time_seconds'].min()
        
        # Filtra valori anomali
        df_clean = df[(df['Dx'] > 0) & (df['Dx'] < 20) & 
                      (df['Dy'] > 0) & (df['Dy'] < 20) & 
                      df['Dx'].notna() & df['Dy'].notna()].copy()
        
        # Ordina per tempo
        df_clean = df_clean.sort_values('time_seconds').reset_index(drop=True)
        
        df_clean['expected_diameter'] = expected_diameter
        df_clean['bar_type'] = bar_type
        
        print(f"✅ Dati validi: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
        print(f"⏱️  Durata: {df_clean['time_seconds'].max():.2f} secondi")
        
        return df_clean
    
    def create_time_windows(self, df):
        """
        Crea finestre temporali (identico al training)
        """
        time_data = df['time_seconds'].values
        dx_data = df['Dx'].values
        dy_data = df['Dy'].values
        expected_diameter = df['expected_diameter'].iloc[0] if 'expected_diameter' in df.columns else None
        bar_type = df['bar_type'].iloc[0] if 'bar_type' in df.columns else 'unknown'
        
        total_duration = time_data.max() - time_data.min()
        step_size = self.window_duration * (1 - self.overlap)
        
        windows = []
        
        start_time = time_data.min()
        while start_time + self.window_duration <= time_data.max():
            end_time = start_time + self.window_duration
            
            mask = (time_data >= start_time) & (time_data < end_time)
            
            if mask.sum() > 10:
                window_data = {
                    'time_start': start_time,
                    'time_end': end_time,
                    'dx_values': dx_data[mask],
                    'dy_values': dy_data[mask],
                    'time_values': time_data[mask],
                    'expected_diameter': expected_diameter,
                    'bar_type': bar_type,
                    'n_points': mask.sum()
                }
                windows.append(window_data)
            
            start_time += step_size
        
        print(f"🔄 Create {len(windows)} finestre di {self.window_duration}s")
        return windows
    
    def extract_window_features(self, window):
        """
        Estrae features da una singola finestra temporale
        VERSIONE OTTIMIZZATA PER GEOMETRIA INTRINSECA DELLE NERVATURE
        Focus: caratteristiche geometriche indipendenti dalla velocità
        """
        dx = window['dx_values']
        dy = window['dy_values']
        time_vals = window['time_values']
        
        features = {}
        
        # ========================================
        # FEATURES GEOMETRICHE FONDAMENTALI
        # Queste sono le PIÙ IMPORTANTI per il diametro equivalente
        # ========================================
        
        features['dx_mean'] = np.mean(dx)
        features['dy_mean'] = np.mean(dy)
        
        # CHIAVE PRIMARIA: Diametro equivalente calcolato direttamente
        features['geometric_diameter'] = np.sqrt(features['dx_mean'] * features['dy_mean'])
        
        # Altre medie geometriche
        features['diameter_harmonic_mean'] = 2 / (1/features['dx_mean'] + 1/features['dy_mean'])
        features['diameter_arithmetic_mean'] = (features['dx_mean'] + features['dy_mean']) / 2
        
        # Area e perimetro equivalenti
        features['area_equivalent'] = np.pi * (features['geometric_diameter'] / 2) ** 2
        features['perimeter_equivalent'] = np.pi * features['geometric_diameter']
        
        # ========================================
        # FEATURES DI FORMA (SHAPE FEATURES)
        # ========================================
        
        features['dx_dy_ratio'] = features['dx_mean'] / features['dy_mean'] if features['dy_mean'] > 0 else 1
        features['ellipticity'] = abs(features['dx_mean'] - features['dy_mean']) / features['geometric_diameter'] if features['geometric_diameter'] > 0 else 0
        features['circularity_index'] = min(features['dx_mean'], features['dy_mean']) / max(features['dx_mean'], features['dy_mean']) if max(features['dx_mean'], features['dy_mean']) > 0 else 1
        
        # Asimmetria forma
        features['dx_dy_diff'] = abs(features['dx_mean'] - features['dy_mean'])
        features['dx_dy_diff_normalized'] = features['dx_dy_diff'] / features['geometric_diameter'] if features['geometric_diameter'] > 0 else 0
        
        # ========================================
        # FEATURES DI VARIABILITÀ (per nervature)
        # CHIAVE: Profondità e dimensione nervature, NON frequenza
        # ========================================
        
        features['dx_std'] = np.std(dx)
        features['dy_std'] = np.std(dy)
        features['dx_min'] = np.min(dx)
        features['dy_min'] = np.min(dy)
        features['dx_max'] = np.max(dx)
        features['dy_max'] = np.max(dy)
        
        # IMPORTANTE: Range = differenza tra picco e valle delle nervature
        features['dx_range'] = features['dx_max'] - features['dx_min']
        features['dy_range'] = features['dy_max'] - features['dy_min']
        
        # Range normalizzato (indipendente dal diametro assoluto)
        features['dx_range_normalized'] = features['dx_range'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_range_normalized'] = features['dy_range'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        features['total_range_normalized'] = (features['dx_range_normalized'] + features['dy_range_normalized']) / 2
        
        # Coefficienti di variazione (profondità nervature relativa)
        features['dx_cv'] = features['dx_std'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_cv'] = features['dy_std'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        features['combined_cv'] = (features['dx_cv'] + features['dy_cv']) / 2
        
        # ========================================
        # FEATURES PERCENTILI (distribuzione valori)
        # ========================================
        
        features['dx_p05'] = np.percentile(dx, 5)
        features['dx_p25'] = np.percentile(dx, 25)
        features['dx_p50'] = np.percentile(dx, 50)  # Mediana
        features['dx_p75'] = np.percentile(dx, 75)
        features['dx_p95'] = np.percentile(dx, 95)
        
        features['dy_p05'] = np.percentile(dy, 5)
        features['dy_p25'] = np.percentile(dy, 25)
        features['dy_p50'] = np.percentile(dy, 50)
        features['dy_p75'] = np.percentile(dy, 75)
        features['dy_p95'] = np.percentile(dy, 95)
        
        # IQR = misura robusta della profondità nervature
        features['dx_iqr'] = features['dx_p75'] - features['dx_p25']
        features['dy_iqr'] = features['dy_p75'] - features['dy_p25']
        features['dx_iqr_normalized'] = features['dx_iqr'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_iqr_normalized'] = features['dy_iqr'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        
        # Range tra percentili estremi (5-95)
        features['dx_p05_p95_range'] = features['dx_p95'] - features['dx_p05']
        features['dy_p05_p95_range'] = features['dy_p95'] - features['dy_p05']
        features['dx_p05_p95_normalized'] = features['dx_p05_p95_range'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_p05_p95_normalized'] = features['dy_p05_p95_range'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        
        # ========================================
        # FEATURES PICCHI/VALLI (GEOMETRIA NERVATURE)
        # Focus: DIMENSIONE non frequenza
        # ========================================
        
        if len(dx) > 10:
            # Soglie adattive per rilevamento picchi/valli
            dx_peak_threshold = features['dx_p75']
            dy_peak_threshold = features['dy_p75']
            dx_valley_threshold = features['dx_p25']
            dy_valley_threshold = features['dy_p25']
            
            # Trova picchi e valli
            dx_peaks, dx_peak_props = signal.find_peaks(dx, height=dx_peak_threshold)
            dy_peaks, dy_peak_props = signal.find_peaks(dy, height=dy_peak_threshold)
            dx_valleys, dx_valley_props = signal.find_peaks(-dx, height=-dx_valley_threshold)
            dy_valleys, dy_valley_props = signal.find_peaks(-dy, height=-dy_valley_threshold)
            
            # CHIAVE: Ampiezza media picco-valle (dimensione nervatura)
            if len(dx_peaks) > 0 and len(dx_valleys) > 0:
                dx_peak_values = dx[dx_peaks]
                dx_valley_values = dx[dx_valleys]
                
                features['dx_peak_mean'] = np.mean(dx_peak_values)
                features['dx_valley_mean'] = np.mean(dx_valley_values)
                features['dx_peak_valley_amplitude'] = features['dx_peak_mean'] - features['dx_valley_mean']
                features['dx_peak_valley_normalized'] = features['dx_peak_valley_amplitude'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
                
                # Variabilità dei picchi (uniformità nervature)
                features['dx_peak_std'] = np.std(dx_peak_values)
                features['dx_valley_std'] = np.std(dx_valley_values)
            else:
                features['dx_peak_mean'] = features['dx_mean']
                features['dx_valley_mean'] = features['dx_mean']
                features['dx_peak_valley_amplitude'] = 0
                features['dx_peak_valley_normalized'] = 0
                features['dx_peak_std'] = 0
                features['dx_valley_std'] = 0
            
            if len(dy_peaks) > 0 and len(dy_valleys) > 0:
                dy_peak_values = dy[dy_peaks]
                dy_valley_values = dy[dy_valleys]
                
                features['dy_peak_mean'] = np.mean(dy_peak_values)
                features['dy_valley_mean'] = np.mean(dy_valley_values)
                features['dy_peak_valley_amplitude'] = features['dy_peak_mean'] - features['dy_valley_mean']
                features['dy_peak_valley_normalized'] = features['dy_peak_valley_amplitude'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
                
                features['dy_peak_std'] = np.std(dy_peak_values)
                features['dy_valley_std'] = np.std(dy_valley_values)
            else:
                features['dy_peak_mean'] = features['dy_mean']
                features['dy_valley_mean'] = features['dy_mean']
                features['dy_peak_valley_amplitude'] = 0
                features['dy_peak_valley_normalized'] = 0
                features['dy_peak_std'] = 0
                features['dy_valley_std'] = 0
            
            # Ampiezza combinata (indicatore principale nervature)
            features['combined_peak_valley_amplitude'] = (features['dx_peak_valley_amplitude'] + features['dy_peak_valley_amplitude']) / 2
            features['combined_peak_valley_normalized'] = (features['dx_peak_valley_normalized'] + features['dy_peak_valley_normalized']) / 2
            
            # Rapporto tra ampiezze Dx e Dy (asimmetria nervature)
            if features['dy_peak_valley_amplitude'] > 0:
                features['peak_valley_asymmetry'] = features['dx_peak_valley_amplitude'] / features['dy_peak_valley_amplitude']
            else:
                features['peak_valley_asymmetry'] = 1.0
                
        else:
            # Valori default se troppi pochi punti
            features['dx_peak_mean'] = features['dx_mean']
            features['dx_valley_mean'] = features['dx_mean']
            features['dx_peak_valley_amplitude'] = 0
            features['dx_peak_valley_normalized'] = 0
            features['dx_peak_std'] = 0
            features['dx_valley_std'] = 0
            features['dy_peak_mean'] = features['dy_mean']
            features['dy_valley_mean'] = features['dy_mean']
            features['dy_peak_valley_amplitude'] = 0
            features['dy_peak_valley_normalized'] = 0
            features['dy_peak_std'] = 0
            features['dy_valley_std'] = 0
            features['combined_peak_valley_amplitude'] = 0
            features['combined_peak_valley_normalized'] = 0
            features['peak_valley_asymmetry'] = 1.0
        
        # ========================================
        # FEATURES STATISTICHE AVANZATE
        # ========================================
        
        if len(dx) > 3:
            from scipy import stats
            
            # Skewness (asimmetria distribuzione - utile per nervature)
            features['dx_skewness'] = stats.skew(dx) if np.std(dx) > 0 else 0
            features['dy_skewness'] = stats.skew(dy) if np.std(dy) > 0 else 0
            
            # Kurtosis (picco distribuzione)
            features['dx_kurtosis'] = stats.kurtosis(dx) if np.std(dx) > 0 else 0
            features['dy_kurtosis'] = stats.kurtosis(dy) if np.std(dy) > 0 else 0
        else:
            features['dx_skewness'] = 0
            features['dy_skewness'] = 0
            features['dx_kurtosis'] = 0
            features['dy_kurtosis'] = 0
        
        # Correlazione Dx-Dy (sincronizzazione nervature)
        if len(dx) > 1 and np.std(dx) > 0 and np.std(dy) > 0:
            features['dx_dy_correlation'] = np.corrcoef(dx, dy)[0, 1]
        else:
            features['dx_dy_correlation'] = 0
        
        # ========================================
        # FEATURES DI TENDENZA (STABILITÀ MISURA)
        # ========================================
        
        if len(dx) > 2:
            # Trend lineare (deriva misura nel tempo)
            time_norm = time_vals - time_vals.min()
            if len(time_norm) > 1:
                dx_trend = np.polyfit(time_norm, dx, 1)[0]
                dy_trend = np.polyfit(time_norm, dy, 1)[0]
            else:
                dx_trend = 0
                dy_trend = 0
            
            features['dx_trend'] = dx_trend
            features['dy_trend'] = dy_trend
            features['dx_trend_normalized'] = dx_trend / features['dx_mean'] if features['dx_mean'] > 0 else 0
            features['dy_trend_normalized'] = dy_trend / features['dy_mean'] if features['dy_mean'] > 0 else 0
            
            # Variazioni consecutive (rugosità misura)
            dx_diff = np.diff(dx)
            dy_diff = np.diff(dy)
            
            features['dx_variation_mean'] = np.mean(np.abs(dx_diff))
            features['dy_variation_mean'] = np.mean(np.abs(dy_diff))
            features['dx_variation_std'] = np.std(dx_diff)
            features['dy_variation_std'] = np.std(dy_diff)
            
            # Normalizzate
            features['dx_variation_normalized'] = features['dx_variation_mean'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
            features['dy_variation_normalized'] = features['dy_variation_mean'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        else:
            features['dx_trend'] = 0
            features['dy_trend'] = 0
            features['dx_trend_normalized'] = 0
            features['dy_trend_normalized'] = 0
            features['dx_variation_mean'] = 0
            features['dy_variation_mean'] = 0
            features['dx_variation_std'] = 0
            features['dy_variation_std'] = 0
            features['dx_variation_normalized'] = 0
            features['dy_variation_normalized'] = 0
        
        # ========================================
        # FEATURES CATEGORICHE
        # ========================================
        
        # Tipo tondino (liscio vs nervato)
        features['is_ribbed'] = 1 if window['bar_type'] == 'nervato' else 0
        
        # Indicatore qualità nervature (basato su range normalizzato)
        # Tondini nervati dovrebbero avere range > 5% del diametro
        features['rib_quality_indicator'] = 1 if features['total_range_normalized'] > 0.05 else 0
        
        # ========================================
        # META FEATURES (QUALITÀ DATI)
        # ========================================
        
        features['n_points'] = len(dx)
        window_duration = time_vals.max() - time_vals.min() if len(time_vals) > 1 else 0
        features['sampling_rate'] = len(dx) / window_duration if window_duration > 0 else 0
        
        return features
        """
        Estrae features da una singola finestra temporale
        VERSIONE MIGLIORATA per regressione continua
        """
        dx = window['dx_values']
        dy = window['dy_values']
        time_vals = window['time_values']
        
        features = {}
        
        # === FEATURES GEOMETRICHE FONDAMENTALI ===
        # Queste sono le più importanti per la regressione continua!
        features['dx_mean'] = np.mean(dx)
        features['dy_mean'] = np.mean(dy)
        
        # CHIAVE: Diametro equivalente calcolato direttamente
        features['geometric_diameter'] = np.sqrt(features['dx_mean'] * features['dy_mean'])
        
        # Features geometriche derivate
        features['diameter_harmonic_mean'] = 2 / (1/features['dx_mean'] + 1/features['dy_mean'])
        features['diameter_arithmetic_mean'] = (features['dx_mean'] + features['dy_mean']) / 2
        features['area_equivalent'] = np.pi * (features['geometric_diameter'] / 2) ** 2
        features['perimeter_equivalent'] = np.pi * features['geometric_diameter']
        
        # Rapporti e forme
        features['dx_dy_ratio'] = features['dx_mean'] / features['dy_mean'] if features['dy_mean'] > 0 else 1
        features['ellipticity'] = abs(features['dx_mean'] - features['dy_mean']) / features['geometric_diameter']
        features['circularity_index'] = min(features['dx_mean'], features['dy_mean']) / max(features['dx_mean'], features['dy_mean'])
        
        # === FEATURES STATISTICHE AVANZATE ===
        features['dx_std'] = np.std(dx)
        features['dy_std'] = np.std(dy)
        features['dx_min'] = np.min(dx)
        features['dy_min'] = np.min(dy)
        features['dx_max'] = np.max(dx)
        features['dy_max'] = np.max(dy)
        features['dx_range'] = features['dx_max'] - features['dx_min']
        features['dy_range'] = features['dy_max'] - features['dy_min']
        
        # Variabilità normalizzata (importante per diversi diametri)
        features['dx_cv'] = features['dx_std'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_cv'] = features['dy_std'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        
        # Correlazione sicura
        if len(dx) > 1 and np.std(dx) > 0 and np.std(dy) > 0:
            features['dx_dy_correlation'] = np.corrcoef(dx, dy)[0, 1]
        else:
            features['dx_dy_correlation'] = 0
        
        # === FEATURES DI SCALA (Scale-invariant) ===
        # Questi rapporti sono indipendenti dal diametro assoluto
        if features['geometric_diameter'] > 0:
            features['dx_normalized'] = features['dx_mean'] / features['geometric_diameter']
            features['dy_normalized'] = features['dy_mean'] / features['geometric_diameter']
            features['dx_std_normalized'] = features['dx_std'] / features['geometric_diameter']
            features['dy_std_normalized'] = features['dy_std'] / features['geometric_diameter']
            features['dx_range_normalized'] = features['dx_range'] / features['geometric_diameter']
            features['dy_range_normalized'] = features['dy_range'] / features['geometric_diameter']
        else:
            features['dx_normalized'] = 1
            features['dy_normalized'] = 1
            features['dx_std_normalized'] = 0
            features['dy_std_normalized'] = 0
            features['dx_range_normalized'] = 0
            features['dy_range_normalized'] = 0
        
        # === FEATURES TEMPORALI ===
        if len(dx) > 2:
            time_norm = time_vals - time_vals.min()
            dx_trend = np.polyfit(time_norm, dx, 1)[0] if len(time_norm) > 1 else 0
            dy_trend = np.polyfit(time_norm, dy, 1)[0] if len(time_norm) > 1 else 0
            features['dx_trend'] = dx_trend
            features['dy_trend'] = dy_trend
            
            # Trend normalizzato
            features['dx_trend_normalized'] = dx_trend / features['dx_mean'] if features['dx_mean'] > 0 else 0
            features['dy_trend_normalized'] = dy_trend / features['dy_mean'] if features['dy_mean'] > 0 else 0
            
            dx_diff = np.diff(dx)
            dy_diff = np.diff(dy)
            features['dx_variation_mean'] = np.mean(np.abs(dx_diff))
            features['dy_variation_mean'] = np.mean(np.abs(dy_diff))
            features['dx_variation_std'] = np.std(dx_diff)
            features['dy_variation_std'] = np.std(dy_diff)
            
            # Variazioni normalizzate
            features['dx_variation_normalized'] = features['dx_variation_mean'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
            features['dy_variation_normalized'] = features['dy_variation_mean'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        else:
            features['dx_trend'] = 0
            features['dy_trend'] = 0
            features['dx_trend_normalized'] = 0
            features['dy_trend_normalized'] = 0
            features['dx_variation_mean'] = 0
            features['dy_variation_mean'] = 0
            features['dx_variation_std'] = 0
            features['dy_variation_std'] = 0
            features['dx_variation_normalized'] = 0
            features['dy_variation_normalized'] = 0
        
        # === FEATURES DI PERIODICITÀ PER NERVATURE ===
        if len(dx) > 20:
            dx_detrended = signal.detrend(dx)
            dy_detrended = signal.detrend(dy)
            
            # FFT
            fft_dx = fft(dx_detrended)
            freqs_dx = fftfreq(len(dx_detrended))
            power_dx = np.abs(fft_dx)
            
            fft_dy = fft(dy_detrended)
            power_dy = np.abs(fft_dy)
            
            positive_mask = freqs_dx > 0
            if positive_mask.sum() > 0:
                features['dx_spectral_energy'] = np.sum(power_dx[positive_mask])
                features['dy_spectral_energy'] = np.sum(power_dy[positive_mask])
                
                # Energia spettrale normalizzata
                total_energy_dx = np.sum(power_dx)
                total_energy_dy = np.sum(power_dy)
                features['dx_spectral_ratio'] = features['dx_spectral_energy'] / total_energy_dx if total_energy_dx > 0 else 0
                features['dy_spectral_ratio'] = features['dy_spectral_energy'] / total_energy_dy if total_energy_dy > 0 else 0
                
                # Frequenza dominante
                max_idx_dx = np.argmax(power_dx[positive_mask])
                max_idx_dy = np.argmax(power_dy[positive_mask])
                features['dx_dominant_freq'] = freqs_dx[positive_mask][max_idx_dx]
                features['dy_dominant_freq'] = freqs_dy[positive_mask][max_idx_dy]
            else:
                features['dx_spectral_energy'] = 0
                features['dy_spectral_energy'] = 0
                features['dx_spectral_ratio'] = 0
                features['dy_spectral_ratio'] = 0
                features['dx_dominant_freq'] = 0
                features['dy_dominant_freq'] = 0
        else:
            features['dx_spectral_energy'] = 0
            features['dy_spectral_energy'] = 0
            features['dx_spectral_ratio'] = 0
            features['dy_spectral_ratio'] = 0
            features['dx_dominant_freq'] = 0
            features['dy_dominant_freq'] = 0
        
        # === FEATURES PER RILEVAMENTO NERVATURE ===
        if len(dx) > 10:
            # Picchi normalizzati rispetto alla scala
            dx_height_threshold = features['dx_mean'] + 0.5 * features['dx_std']
            dy_height_threshold = features['dy_mean'] + 0.5 * features['dy_std']
            
            dx_peaks, _ = signal.find_peaks(dx, height=dx_height_threshold)
            dy_peaks, _ = signal.find_peaks(dy, height=dy_height_threshold)
            
            dx_valley_threshold = features['dx_mean'] - 0.5 * features['dx_std']
            dy_valley_threshold = features['dy_mean'] - 0.5 * features['dy_std']
            
            dx_valleys, _ = signal.find_peaks(-dx, height=-dx_valley_threshold)
            dy_valleys, _ = signal.find_peaks(-dy, height=-dy_valley_threshold)
            
            features['dx_n_peaks'] = len(dx_peaks)
            features['dy_n_peaks'] = len(dy_peaks)
            features['dx_n_valleys'] = len(dx_valleys)
            features['dy_n_valleys'] = len(dy_valleys)
            
            # Densità normalizzata
            window_duration = time_vals.max() - time_vals.min()
            features['dx_peak_density'] = len(dx_peaks) / window_duration if window_duration > 0 else 0
            features['dy_peak_density'] = len(dy_peaks) / window_duration if window_duration > 0 else 0
            
            # Ampiezza normalizzata
            if len(dx_peaks) > 0 and len(dx_valleys) > 0:
                peak_valley_amp = np.mean(dx[dx_peaks]) - np.mean(dx[dx_valleys])
                features['dx_peak_valley_amplitude'] = peak_valley_amp
                features['dx_peak_valley_normalized'] = peak_valley_amp / features['dx_mean'] if features['dx_mean'] > 0 else 0
            else:
                features['dx_peak_valley_amplitude'] = 0
                features['dx_peak_valley_normalized'] = 0
                
            if len(dy_peaks) > 0 and len(dy_valleys) > 0:
                peak_valley_amp = np.mean(dy[dy_peaks]) - np.mean(dy[dy_valleys])
                features['dy_peak_valley_amplitude'] = peak_valley_amp
                features['dy_peak_valley_normalized'] = peak_valley_amp / features['dy_mean'] if features['dy_mean'] > 0 else 0
            else:
                features['dy_peak_valley_amplitude'] = 0
                features['dy_peak_valley_normalized'] = 0
        else:
            features['dx_n_peaks'] = 0
            features['dy_n_peaks'] = 0
            features['dx_n_valleys'] = 0
            features['dy_n_valleys'] = 0
            features['dx_peak_density'] = 0
            features['dy_peak_density'] = 0
            features['dx_peak_valley_amplitude'] = 0
            features['dy_peak_valley_amplitude'] = 0
            features['dx_peak_valley_normalized'] = 0
            features['dy_peak_valley_normalized'] = 0
        
        # === FEATURES STATISTICHE AVANZATE ===
        if len(dx) > 3:
            from scipy import stats
            features['dx_skewness'] = stats.skew(dx) if np.std(dx) > 0 else 0
            features['dy_skewness'] = stats.skew(dy) if np.std(dy) > 0 else 0
            features['dx_kurtosis'] = stats.kurtosis(dx) if np.std(dx) > 0 else 0
            features['dy_kurtosis'] = stats.kurtosis(dy) if np.std(dy) > 0 else 0
        else:
            features['dx_skewness'] = 0
            features['dy_skewness'] = 0
            features['dx_kurtosis'] = 0
            features['dy_kurtosis'] = 0
        
        # Percentili
        features['dx_p25'] = np.percentile(dx, 25)
        features['dx_p75'] = np.percentile(dx, 75)
        features['dy_p25'] = np.percentile(dy, 25)
        features['dy_p75'] = np.percentile(dy, 75)
        features['dx_iqr'] = features['dx_p75'] - features['dx_p25']
        features['dy_iqr'] = features['dy_p75'] - features['dy_p25']
        
        # IQR normalizzato
        features['dx_iqr_normalized'] = features['dx_iqr'] / features['dx_mean'] if features['dx_mean'] > 0 else 0
        features['dy_iqr_normalized'] = features['dy_iqr'] / features['dy_mean'] if features['dy_mean'] > 0 else 0
        
        # === FEATURES TIPO TONDINO (CATEGORICAL) ===
        features['is_ribbed'] = 1 if window['bar_type'] == 'nervato' else 0
        
        # === META FEATURES ===
        features['n_points'] = len(dx)
        features['sampling_rate'] = len(dx) / window_duration if window_duration > 0 else 0
        
        return features
    
    def predict_from_windows(self, windows):
        """
        Effettua predizioni su una lista di finestre temporali
        """
        if not windows:
            return np.array([]), np.array([])
        
        # Estrai features da tutte le finestre
        features_list = []
        for window in windows:
            features = self.extract_window_features(window)
            features_list.append(features)
        
        # Converti in DataFrame
        X = pd.DataFrame(features_list)
        
        # Assicurati che tutte le features del training siano presenti
        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = 0
        
        # Riordina le colonne come nel training
        X = X[self.feature_names]
        
        # Predizione
        predictions = self.best_model.predict(X)
        
        return predictions, X
    
    def test_single_file(self, file_path, expected_diameter=None, bar_type='unknown', plot=True):
        """
        Testa il modello su un singolo file
        """
        print(f"\n🧪 === TEST SU {Path(file_path).name} ===")
        
        try:
            # Carica dati
            df = self.load_and_clean_data(file_path, expected_diameter, bar_type)
            
            # Crea finestre temporali
            windows = self.create_time_windows(df)
            
            if not windows:
                print("❌ Nessuna finestra temporale creata - file troppo corto")
                return None
            
            # Predizioni
            predictions, features_df = self.predict_from_windows(windows)
            
            # Statistiche predizioni
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            pred_min = np.min(predictions)
            pred_max = np.max(predictions)
            
            print(f"\n📊 RISULTATI PREDIZIONE:")
            print(f"   🎯 Diametro stimato: {pred_mean:.4f} ± {pred_std:.4f} mm")
            print(f"   📈 Range: [{pred_min:.4f}, {pred_max:.4f}] mm")
            print(f"   📦 Basato su {len(predictions)} finestre temporali")
            
            if expected_diameter:
                error = abs(pred_mean - expected_diameter)
                relative_error = (error / expected_diameter) * 100
                print(f"   🎲 Diametro atteso: {expected_diameter:.1f} mm")
                print(f"   ❌ Errore assoluto: {error:.4f} mm")
                print(f"   📊 Errore relativo: {relative_error:.2f}%")
                
                # Valutazione qualitativa
                if relative_error < 0.1:
                    print(f"   ✅ Predizione ECCELLENTE (< 0.1%)")
                elif relative_error < 0.5:
                    print(f"   ✅ Predizione BUONA (< 0.5%)")
                elif relative_error < 1:
                    print(f"   ⚠️  Predizione ACCETTABILE (< 1%)")
                else:
                    print(f"   ❌ Predizione SCADENTE (> 1%)")
            
            # Analisi stabilità predizioni
            pred_cv = (pred_std / pred_mean) * 100 if pred_mean > 0 else 0
            print(f"   📏 Coefficiente di variazione: {pred_cv:.2f}%")
            
            if pred_cv < 1:
                print(f"   ✅ Predizioni molto STABILI")
            elif pred_cv < 3:
                print(f"   ✅ Predizioni STABILI")
            elif pred_cv < 5:
                print(f"   ⚠️  Predizioni moderatamente stabili")
            else:
                print(f"   ❌ Predizioni INSTABILI")
            
            # Confronto con calcolo geometrico diretto
            geometric_mean = np.sqrt(df['Dx'].mean() * df['Dy'].mean())
            print(f"   🧮 Calcolo geometrico diretto: {geometric_mean:.4f} mm")
            print(f"   🔄 Differenza ML vs geometrico: {abs(pred_mean - geometric_mean):.4f} mm")
            
            if plot:
                self._plot_single_file_results(df, windows, predictions, file_path)
            
            return {
                'file_path': file_path,
                'predictions': predictions,
                'mean_prediction': pred_mean,
                'std_prediction': pred_std,
                'expected_diameter': expected_diameter,
                'error': error if expected_diameter else None,
                'relative_error': relative_error if expected_diameter else None,
                'geometric_mean': geometric_mean,
                'n_windows': len(windows),
                'cv_percent': pred_cv
            }
            
        except Exception as e:
            print(f"❌ Errore durante il test: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_single_file_results(self, df, windows, predictions, file_path):
        """
        Visualizza i risultati per un singolo file
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Analisi Completa - {Path(file_path).name}', fontsize=16)
        
        # Plot 1: Serie temporali originali
        axes[0,0].plot(df['time_seconds'], df['Dx'], 'b-', alpha=0.7, label='Dx')
        axes[0,0].plot(df['time_seconds'], df['Dy'], 'r-', alpha=0.7, label='Dy')
        axes[0,0].set_title('Misure Temporali Originali')
        axes[0,0].set_xlabel('Tempo (s)')
        axes[0,0].set_ylabel('Diametro (mm)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Predizioni per finestra
        window_times = [(w['time_start'] + w['time_end']) / 2 for w in windows]
        axes[0,1].plot(window_times, predictions, 'go-', markersize=6, linewidth=2)
        axes[0,1].axhline(y=np.mean(predictions), color='r', linestyle='--', 
                         label=f'Media: {np.mean(predictions):.3f}mm')
        axes[0,1].fill_between(window_times, 
                              np.mean(predictions) - np.std(predictions),
                              np.mean(predictions) + np.std(predictions),
                              alpha=0.3, color='gray', label='±1σ')
        axes[0,1].set_title('Predizioni per Finestra Temporale')
        axes[0,1].set_xlabel('Tempo (s)')
        axes[0,1].set_ylabel('Diametro Equivalente (mm)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Istogramma predizioni
        axes[0,2].hist(predictions, bins=20, alpha=0.7, density=True, color='green')
        axes[0,2].axvline(np.mean(predictions), color='red', linestyle='--', linewidth=2)
        axes[0,2].set_title('Distribuzione Predizioni')
        axes[0,2].set_xlabel('Diametro Equivalente (mm)')
        axes[0,2].set_ylabel('Densità')
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Scatter Dx vs Dy colorato per tempo
        scatter = axes[1,0].scatter(df['Dx'], df['Dy'], c=df['time_seconds'], 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[1,0].set_title('Relazione Dx vs Dy nel Tempo')
        axes[1,0].set_xlabel('Dx (mm)')
        axes[1,0].set_ylabel('Dy (mm)')
        plt.colorbar(scatter, ax=axes[1,0], label='Tempo (s)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Variazione nel tempo
        dx_variation = np.abs(np.diff(df['Dx']))
        dy_variation = np.abs(np.diff(df['Dy']))
        time_diff = df['time_seconds'].iloc[1:].values
        
        axes[1,1].plot(time_diff, dx_variation, 'b-', alpha=0.7, label='Variazione Dx')
        axes[1,1].plot(time_diff, dy_variation, 'r-', alpha=0.7, label='Variazione Dy')
        axes[1,1].set_title('Variazioni Temporali')
        axes[1,1].set_xlabel('Tempo (s)')
        axes[1,1].set_ylabel('Variazione Assoluta (mm)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Box plot per stabilità
        pred_data = [predictions]
        geometric_data = [np.sqrt(df['Dx'] * df['Dy'])]
        
        axes[1,2].boxplot([predictions, np.sqrt(df['Dx'] * df['Dy'])], 
                         labels=['ML Prediction', 'Geometric'])
        axes[1,2].set_title('Confronto Stabilità')
        axes[1,2].set_ylabel('Diametro Equivalente (mm)')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def batch_test(self, test_files_or_config):
        """
        Testa il modello su più file
        """
        print("🧪 === TEST BATCH SU MULTIPLI FILE ===\n")
        
        # Gestisci input diversi
        if isinstance(test_files_or_config, str):
            # È un file di configurazione
            dm = TondinoDataManager()
            if dm.load_config(test_files_or_config):
                test_files = dm.get_file_list_for_training()
            else:
                print(f"❌ Impossibile caricare configurazione: {test_files_or_config}")
                return []
        else:
            # È una lista di file
            test_files = test_files_or_config
        
        results = []
        successful_tests = 0
        
        for i, file_info in enumerate(test_files, 1):
            print(f"\n[{i}/{len(test_files)}] ", end="")
            
            if len(file_info) == 3:
                file_path, expected_diameter, bar_type = file_info
            else:
                file_path = file_info
                expected_diameter = None
                bar_type = 'unknown'
            
            if os.path.exists(file_path):
                result = self.test_single_file(file_path, expected_diameter, bar_type, plot=False)
                if result:
                    results.append(result)
                    successful_tests += 1
            else:
                print(f"❌ File non trovato: {file_path}")
        
        # Riassunto batch
        if results:
            print(f"\n📊 === RIASSUNTO BATCH TEST ===")
            print(f"✅ Test riusciti: {successful_tests}/{len(test_files)}")
            
            # Statistiche aggregate
            all_errors = [r['relative_error'] for r in results if r['relative_error'] is not None]
            all_predictions = [r['mean_prediction'] for r in results]
            all_cvs = [r['cv_percent'] for r in results]
            
            if all_errors:
                print(f"📈 Errore relativo medio: {np.mean(all_errors):.2f}% (±{np.std(all_errors):.2f}%)")
                print(f"📉 Errore relativo mediano: {np.median(all_errors):.2f}%")
                print(f"📊 Range errori: [{np.min(all_errors):.2f}%, {np.max(all_errors):.2f}%]")
            
            print(f"🎯 Stabilità media (CV): {np.mean(all_cvs):.2f}%")
            
            # Risultati per categoria
            categories = {}
            for result in results:
                if result['expected_diameter']:
                    diameter = result['expected_diameter']
                    if diameter not in categories:
                        categories[diameter] = []
                    categories[diameter].append(result)
            
            print(f"\n📋 Risultati per categoria:")
            for diameter, cat_results in categories.items():
                errors = [r['relative_error'] for r in cat_results if r['relative_error'] is not None]
                if errors:
                    print(f"   D{diameter}mm: {len(cat_results)} file, errore medio {np.mean(errors):.2f}%")
        
        return results
    
    def interactive_prediction(self):
        """
        Modalità predizione interattiva per singoli file
        """
        print("\n🎛️  === MODALITÀ PREDIZIONE INTERATTIVA ===")
        print("Inserisci il percorso di un file CSV per ottenere una predizione")
        print("Digita 'quit' per uscire\n")
        
        while True:
            try:
                file_input = input("📁 Percorso file CSV: ").strip()
                if file_input.lower() == 'quit':
                    break
                
                if not os.path.exists(file_input):
                    print(f"❌ File non trovato: {file_input}")
                    continue
                
                # Chiedi informazioni opzionali
                diameter_input = input("🎯 Diametro atteso (mm, oppure ENTER per saltare): ").strip()
                expected_diameter = float(diameter_input) if diameter_input else None
                
                type_input = input("🔧 Tipo tondino (liscio/nervato, oppure ENTER per auto): ").strip().lower()
                bar_type = type_input if type_input in ['liscio', 'nervato'] else 'unknown'
                
                # Esegui predizione
                result = self.test_single_file(file_input, expected_diameter, bar_type, plot=True)
                
                if result:
                    print(f"\n✅ Predizione completata!")
                else:
                    print(f"\n❌ Predizione fallita")
                
                print("\n" + "="*50)
                
            except ValueError:
                print("❌ Errore: inserisci un numero valido per il diametro\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Errore: {e}\n")


def main():
    """
    Funzione principale per il testing
    """
    parser = argparse.ArgumentParser(description='Test del modello di stima del diametro equivalente')
    parser.add_argument('--model', default='timeseries_diameter_estimator.pkl', 
                       help='Percorso del modello salvato')
    parser.add_argument('--file', help='File CSV singolo da testare')
    parser.add_argument('--diameter', type=float, help='Diametro atteso per il file singolo')
    parser.add_argument('--type', choices=['liscio', 'nervato'], default='unknown',
                       help='Tipo di tondino')
    parser.add_argument('--interactive', action='store_true',
                       help='Modalità predizione interattiva')
    parser.add_argument('--batch', help='Test batch - percorso file config o directory')
    parser.add_argument('--config', help='File di configurazione per test batch')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disabilita i grafici')
    
    args = parser.parse_args()
    
    try:
        # Inizializza il tester
        print("🚀 INIZIALIZZAZIONE TESTER SERIE TEMPORALI")
        print("=" * 50)
        tester = TimeSeriesDiameterTester(args.model)
        
        if args.interactive:
            # Modalità interattiva
            tester.interactive_prediction()
            
        elif args.file:
            # Test su file singolo
            if not os.path.exists(args.file):
                print(f"❌ Errore: File {args.file} non trovato")
                return
            
            plot = not args.no_plot
            result = tester.test_single_file(args.file, args.diameter, args.type, plot=plot)
            
            if result:
                print(f"\n🎉 Test completato con successo!")
            else:
                print(f"\n❌ Test fallito")
                
        elif args.batch or args.config:
            # Test batch
            config_file = args.config or args.batch
            
            if config_file and os.path.exists(config_file):
                # Usa file di configurazione
                results = tester.batch_test(config_file)
            elif args.batch and os.path.isdir(args.batch):
                # Usa directory - auto rileva file
                print(f"🔍 Auto-rilevamento file in {args.batch}...")
                dm = TondinoDataManager(args.batch)
                detected = dm.auto_detect_files()
                
                if detected:
                    test_files = [(info['file_path'], info['diameter'], info['type']) for info in detected]
                    results = tester.batch_test(test_files)
                else:
                    print(f"❌ Nessun file rilevato in {args.batch}")
            else:
                print(f"❌ Configurazione o directory non trovata: {config_file or args.batch}")
                return
                
        else:
            # Default: mostra help e test interattivo se non ci sono argomenti
            print("📖 Nessuna opzione specificata. Opzioni disponibili:")
            print("")
            print("🔧 COMANDI PRINCIPALI:")
            print("  --interactive          Modalità interattiva")
            print("  --file FILE            Test su singolo file")
            print("  --batch CONFIG         Test batch da configurazione")
            print("")
            print("🎯 ESEMPI:")
            print("  python test_timeseries_diameter_estimator.py --interactive")
            print("  python test_timeseries_diameter_estimator.py --file D8_test.csv --diameter 8")
            print("  python test_timeseries_diameter_estimator.py --batch data_config.json")
            print("  python test_timeseries_diameter_estimator.py --batch data/")
            print("")
            
            # Chiedi se vuole modalità interattiva
            if input("🎛️  Avviare modalità interattiva? (y/n): ").lower() == 'y':
                tester.interactive_prediction()
            
    except FileNotFoundError as e:
        print(f"❌ Errore: {e}")
        print(f"💡 Assicurati di aver fatto il training prima:")
        print(f"   python train_diameter_estimator.py")
        
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
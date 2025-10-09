def train_models(self, X, y, test_size=0.2):
        """
        Addestra diversi modelli per stimare il diametro
        VERSIONE MIGLIORATA per regressione continua
        """
        print("Divisione train/test...")
        
        # Verifica se possiamo fare stratificazione
        unique_vals, counts = np.unique(y, return_counts=True)
        min_samples_per_class = np.min(counts)
        
        if min_samples_per_class >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print("Utilizzata stratificazione per il train/test split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print("Stratificazione non possibile - utilizzato split casuale")
        
        # === MODELLI OTTIMIZZATI PER REGRESSIONE CONTINUA ===
        models_to_test = {
            # Random Forest con parametri ottimizzati per regressione
            'RandomForest_Optimized': RandomForestRegressor(
                n_estimators=300,           # Più alberi per smoothness
                max_depth=None,            # Profondità illimitata
                min_samples_split=5,       # Divisioni più granulari
                min_samples_leaf=2,        # Foglie più piccole
                max_features='sqrt',       # Feature selection
                bootstrap=True,
                random_state=42,
                n_jobs=-1                  # Parallelizzazione
            ),
            
            # Gradient Boosting ottimizzato
            'GradientBoosting_Optimized': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,             # Bagging per generalizzazione
                random_state=42
            ),
            
            # Extra Trees (più smooth del Random Forest)
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=False,           # No bootstrap per Extra Trees
                random_state=42,
                n_jobs=-1
            ),
            
            # Modelli lineari per catturare relazioni dirette
            'Ridge_Polynomial': Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=1.0))
            ]),
            
            'ElasticNet': Pipeline([
                ('scaler', StandardScaler()),
                ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
            ]),
            
            # SVR con kernel più adatti
            'SVR_RBF': Pipeline([
                ('scaler', RobustScaler()),
                ('svr', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1))
            ]),
            
            'SVR_Polynomial': Pipeline([
                ('scaler', RobustScaler()), 
                ('svr', SVR(kernel='poly', degree=2, C=10.0, epsilon=0.1))
            ]),
            
            # Rete neurale più profonda
            'MLP_Deep': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=(150, 100, 50, 25),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                ))
            ]),
            
            # Modello ensemble personalizzato
            'Voting_Regressor': VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('svr', Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR(kernel='rbf', C=10.0))
                ]))
            ])
        }
        
        best_score = float('-inf')
        best_model_name = None
        
        print("\nTraining e valutazione dei modelli per regressione continua...")
        for name, model in models_to_test.items():
            print(f"\nTraining {name}...")
            
            try:
                # Training
                model.fit(X_train, y_train)
                
                # Predizione
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metriche
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # Test di continuità (importante!)
                # Verifica se il modello può predire valori intermedi
                unique_preds = np.unique(np.round(y_pred_test, 2))
                continuity_score = len(unique_preds) / len(y_pred_test)
                
                print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"{name} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
                print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                print(f"{name} - Continuity Score: {continuity_score:.4f} (higher = better)")
                
                # Salva il modello
                self.models[name] = model
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    print(f"{name} - CV R² medio: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                except:
                    print(f"{name} - CV non possibile")
                
                # Penalizza modelli che non riescono a fare interpolazione
                adjusted_score = test_r2 * continuity_score
                
                # Seleziona il miglior modello basato su score aggiustato
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_model_name = name
                    
            except Exception as e:
                print(f"{name} - ERRORE durante training: {e}")
                continue
        
        print(f"\nMiglior modello: {best_model_name} (Adjusted Score = {best_score:.4f})")
        
        # Salva informazioni sul modello migliore
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        # Feature importance per modelli tree-based
        if best_model_name in ['RandomForest_Optimized', 'GradientBoosting_Optimized', 'ExtraTrees']:
            try:
                if hasattr(self.best_model, 'feature_importances_'):
                    importances = self.best_model.feature_importances_
                elif hasattr(self.best_model, 'named_steps') and hasattr(self.best_model.named_steps.get('model', self.best_model), 'feature_importances_'):
                    importances = self.best_model.named_steps['model'].feature_importances_
                else:
                    importances = None
                
                if importances is not None:
                    self.feature_importance = dict(zip(self.feature_names, importances))
                    
                    # Mostra le feature più importanti
                    sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                    print(f"\nTop 15 feature più importanti per {best_model_name}:")
                    for feat, imp in sorted_features[:15]:
                        print(f"  {feat}: {imp:.4f}")
            except Exception as e:
                print(f"Impossibile estrarre feature importance: {e}")
        
        # Test finale di continuità
        y_pred_final = self.best_model.predict(X_test)
        print(f"\n=== ANALISI CONTINUITÀ MODELLO MIGLIORE ===")
        print(f"Range predizioni: [{y_pred_final.min():.3f}, {y_pred_final.max():.3f}]")
        print(f"Valori unici (arrotondati a 0.1mm): {len(np.unique(np.round(y_pred_final, 1)))}")
        print(f"Standard deviation predizioni: {np.std(y_pred_final):.4f}")
        
        return X_test, y_test, y_pred_final#!/usr/bin/env python3
"""
Script per il training del modello di stima del diametro equivalente dei tondini
con analisi di serie temporali
Autore: Assistente AI
Data: 2025

Questo script implementa un algoritmo di machine learning per stimare il diametro 
equivalente di tondini di ferro analizzando finestre temporali di misure Dx e Dy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
from scipy import signal
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class TimeSeriesDiameterEstimator:
    def __init__(self, window_duration=1.0, overlap=0.5):
        """
        Inizializza l'estimatore per serie temporali
        
        Args:
            window_duration: durata della finestra temporale in secondi
            overlap: sovrapposizione tra finestre (0-1)
        """
        self.window_duration = window_duration
        self.overlap = overlap
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.feature_names = []
        
    def parse_time_column(self, time_str):
        """
        Converte la stringa tempo in secondi relativi
        """
        try:
            # Formato: "15:59:59.080"
            if ':' in str(time_str):
                time_parts = str(time_str).split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                # Gestisci millisecondi con lunghezza variabile
                if len(seconds_parts) > 1:
                    milliseconds_str = seconds_parts[1].ljust(3, '0')[:3]  # Pad o tronca a 3 cifre
                    milliseconds = int(milliseconds_str)
                else:
                    milliseconds = 0
                
                total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                return total_seconds
            else:
                # Se è già un numero, restituiscilo
                return float(time_str)
        except Exception as e:
            print(f"Errore parsing tempo '{time_str}': {e}")
            return None
    
    def load_and_clean_data(self, file_path, true_diameter, bar_type='unknown'):
        """
        Carica e pulisce i dati da un file CSV
        """
        print(f"Caricamento dati da {file_path}...")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Converti colonna tempo in secondi relativi
        df['time_seconds'] = df['Tempo'].apply(self.parse_time_column)
        df = df.dropna(subset=['time_seconds'])
        
        # Normalizza il tempo (inizia da 0)
        df['time_seconds'] = df['time_seconds'] - df['time_seconds'].min()
        
        # Filtra valori anomali
        df_clean = df[(df['Dx'] > 0) & (df['Dx'] < 20) & 
                      (df['Dy'] > 0) & (df['Dy'] < 20) & 
                      df['Dx'].notna() & df['Dy'].notna()].copy()
        
        # Ordina per tempo
        df_clean = df_clean.sort_values('time_seconds').reset_index(drop=True)
        
        # Aggiungi informazioni
        df_clean['true_diameter'] = true_diameter
        df_clean['bar_type'] = bar_type
        
        print(f"Dati validi: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
        print(f"Durata totale: {df_clean['time_seconds'].max():.2f} secondi")
        
        return df_clean
    
    def create_time_windows(self, df):
        """
        Crea finestre temporali sovrapposte dai dati
        """
        time_data = df['time_seconds'].values
        dx_data = df['Dx'].values
        dy_data = df['Dy'].values
        true_diameter = df['true_diameter'].iloc[0]
        bar_type = df['bar_type'].iloc[0]
        
        # Calcola parametri finestre
        total_duration = time_data.max() - time_data.min()
        step_size = self.window_duration * (1 - self.overlap)
        
        windows = []
        
        start_time = time_data.min()
        while start_time + self.window_duration <= time_data.max():
            end_time = start_time + self.window_duration
            
            # Trova i punti in questa finestra
            mask = (time_data >= start_time) & (time_data < end_time)
            
            if mask.sum() > 10:  # Minimo 10 punti per finestra
                window_data = {
                    'time_start': start_time,
                    'time_end': end_time,
                    'dx_values': dx_data[mask],
                    'dy_values': dy_data[mask],
                    'time_values': time_data[mask],
                    'true_diameter': true_diameter,
                    'bar_type': bar_type,
                    'n_points': mask.sum()
                }
                windows.append(window_data)
            
            start_time += step_size
        
        print(f"Create {len(windows)} finestre di {self.window_duration}s con overlap {self.overlap}")
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
    
    def prepare_dataset(self, data_files):
        """
        Prepara il dataset completo da più file con finestre temporali
        
        Args:
            data_files: lista di tuple (file_path, true_diameter, bar_type)
        """
        all_windows = []
        
        for file_path, true_diameter, bar_type in data_files:
            if not os.path.exists(file_path):
                print(f"File non trovato: {file_path}")
                continue
                
            df = self.load_and_clean_data(file_path, true_diameter, bar_type)
            windows = self.create_time_windows(df)
            all_windows.extend(windows)
        
        print(f"\nTotale finestre create: {len(all_windows)}")
        
        # Estrai features da tutte le finestre
        features_list = []
        targets = []
        
        for window in all_windows:
            features = self.extract_window_features(window)
            features_list.append(features)
            targets.append(window['true_diameter'])
        
        # Converti in DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(targets)
        
        # Salva i nomi delle features
        self.feature_names = X.columns.tolist()
        
        print(f"Dataset preparato: {X.shape[0]} finestre, {X.shape[1]} features")
        print(f"Distribuzione target: {np.unique(y, return_counts=True)}")
        
        return X, y, all_windows
    
    def train_models(self, X, y, test_size=0.2):
        """
        Addestra diversi modelli per stimare il diametro
        """
        print("Divisione train/test...")
        
        # Verifica se possiamo fare stratificazione
        unique_vals, counts = np.unique(y, return_counts=True)
        min_samples_per_class = np.min(counts)
        
        if min_samples_per_class >= 2:
            # Possiamo stratificare
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print("Utilizzata stratificazione per il train/test split")
        else:
            # Non possiamo stratificare - troppi pochi campioni per classe
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            print("Stratificazione non possibile - utilizzato split casuale")
        
        # Definisci i modelli da testare
        models_to_test = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=15, random_state=42),
        }
        
        best_score = float('-inf')
        best_model_name = None
        
        print("\nTraining e valutazione dei modelli...")
        for name, model in models_to_test.items():
            print(f"\nTraining {name}...")
            
            # Crea pipeline con scaling per modelli che ne hanno bisogno
            if name in ['SVR', 'MLP', 'Ridge']:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', model)
                ])
            else:
                pipeline = Pipeline([
                    ('model', model)
                ])
            
            # Training
            pipeline.fit(X_train, y_train)
            
            # Predizione
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Metriche
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            print(f"{name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            print(f"{name} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            print(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            
            # Salva il modello
            self.models[name] = pipeline
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            print(f"{name} - CV R² medio: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            
            # Seleziona il miglior modello
            if test_r2 > best_score:
                best_score = test_r2
                best_model_name = name
        
        print(f"\nMiglior modello: {best_model_name} (Test R² = {best_score:.4f})")
        
        # Salva informazioni sul modello migliore
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        # Feature importance
        if best_model_name in ['RandomForest', 'GradientBoosting']:
            if hasattr(self.best_model.named_steps['model'], 'feature_importances_'):
                importances = self.best_model.named_steps['model'].feature_importances_
                self.feature_importance = dict(zip(self.feature_names, importances))
                
                # Mostra le feature più importanti
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                print(f"\nTop 15 feature più importanti per {best_model_name}:")
                for feat, imp in sorted_features[:15]:
                    print(f"  {feat}: {imp:.4f}")
        
        return X_test, y_test, self.best_model.predict(X_test)
    
    def save_model(self, model_path='timeseries_diameter_estimator.pkl'):
        """
        Salva il modello addestrato con configurazioni
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'window_duration': self.window_duration,
            'overlap': self.overlap,
            'models': self.models
        }
        joblib.dump(model_data, model_path)
        print(f"Modello salvato in {model_path}")
    
    def plot_results(self, y_true, y_pred, title="Predizioni vs Valori Reali"):
        """
        Visualizza i risultati del modello
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Scatter plot predizioni vs valori reali
        plt.subplot(2, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.7, s=50)
        
        # Linea ideale
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideale')
        
        plt.xlabel('Diametro Reale (mm)')
        plt.ylabel('Diametro Predetto (mm)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metriche
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f} mm\nRMSE = {rmse:.4f} mm', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Residui
        plt.subplot(2, 3, 2)
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.7, s=50)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Diametro Predetto (mm)')
        plt.ylabel('Residui (mm)')
        plt.title('Grafico dei Residui')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Distribuzione per diametro
        plt.subplot(2, 3, 3)
        unique_diameters = np.unique(y_true)
        for diameter in unique_diameters:
            mask = y_true == diameter
            plt.hist(y_pred[mask], alpha=0.7, label=f'D{diameter}mm', bins=20, density=True)
        plt.xlabel('Predizione (mm)')
        plt.ylabel('Densità')
        plt.title('Distribuzione Predizioni per Diametro')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Box plot errori per diametro
        plt.subplot(2, 3, 4)
        error_data = []
        error_labels = []
        for diameter in unique_diameters:
            mask = y_true == diameter
            errors = np.abs(y_pred[mask] - y_true[mask])
            error_data.append(errors)
            error_labels.append(f'D{diameter}')
        
        plt.boxplot(error_data, labels=error_labels)
        plt.ylabel('Errore Assoluto (mm)')
        plt.title('Distribuzione Errori per Diametro')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Feature importance (se disponibile)
        plt.subplot(2, 3, 5)
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importanza')
            plt.title('Top 10 Feature Importanti')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Confusion matrix per classificazione diametri
        plt.subplot(2, 3, 6)
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Arrotonda le predizioni al diametro più vicino per la matrice di confusione
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        cm = confusion_matrix(y_true_rounded, y_pred_rounded)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_true_rounded),
                    yticklabels=np.unique(y_true_rounded))
        plt.xlabel('Predizione (mm)')
        plt.ylabel('Reale (mm)')
        plt.title('Matrice di Confusione')
        
        plt.tight_layout()
        plt.show()


from data_manager import TondinoDataManager

def main():
    """
    Funzione principale per il training del modello su serie temporali
    """
    print("=== TRAINING MODELLO STIMA DIAMETRO CON SERIE TEMPORALI ===\n")
    
    # Configurazione finestre temporali
    window_duration = 1.0  # 1 secondo per finestra
    overlap = 0.5  # 50% di sovrapposizione
    
    # Inizializza il modello
    estimator = TimeSeriesDiameterEstimator(window_duration=window_duration, overlap=overlap)
    
    # ========== GESTIONE DATI ==========
    print("🗂️  GESTIONE DATI")
    print("-" * 30)
    
    # Opzione 1: Usa data manager per molti file
    dm = TondinoDataManager("data")
    
    # Prova a caricare configurazione esistente
    if dm.load_config("data_config.json"):
        # Usa configurazione esistente
        data_files = dm.get_file_list_for_training()
        print(f"✅ Utilizzando configurazione esistente con {len(data_files)} file")
        
    else:
        print("⚠️  Configurazione non trovata")
        
        # Prova auto-rilevamento
        print("🔍 Tentativo auto-rilevamento...")
        detected = dm.auto_detect_files()
        
        if detected:
            print(f"🎯 Rilevati {len(detected)} file automaticamente")
            create_config = input("Creare configurazione da rilevamento? (y/n): ").lower() == 'y'
            
            if create_config:
                dm.create_config_from_detection()
                data_files = dm.get_file_list_for_training()
            else:
                data_files = [(info['file_path'], info['diameter'], info['type']) for info in detected]
        else:
            # Fallback: file di esempio nella directory corrente
            print("📁 Utilizzo file di esempio nella directory corrente...")
            data_files = [
                ('D8_liscio_1.csv', 8.0, 'liscio'),
                ('D8_creste_1.csv', 8.0, 'nervato'), 
                ('D6_creste_1.csv', 6.0, 'nervato')
            ]
    
    # Verifica che abbiamo file da processare
    if not data_files:
        print("❌ Nessun file dati trovato!")
        print("\n💡 COME PROCEDERE:")
        print("1. Crea una cartella 'data/' e organizza i file")
        print("2. Oppure esegui: python data_manager.py per configurazione interattiva")
        print("3. Oppure metti i file CSV nella directory corrente")
        return
    
    # Filtra file esistenti
    existing_files = [(f, d, t) for f, d, t in data_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"❌ Nessuno dei {len(data_files)} file specificati esiste!")
        print("File richiesti:")
        for f, d, t in data_files:
            print(f"   - {f}")
        return
    
    if len(existing_files) < len(data_files):
        missing = len(data_files) - len(existing_files)
        print(f"⚠️  {missing} file mancanti, utilizzo {len(existing_files)} file disponibili")
    
    data_files = existing_files
    
    try:
        # Prepara il dataset con finestre temporali
        print("Preparazione del dataset con finestre temporali...")
        X, y, windows = estimator.prepare_dataset(data_files)
        
        print(f"Dataset preparato: {X.shape[0]} finestre temporali, {X.shape[1]} features")
        print(f"Diametri nel dataset: {np.unique(y)}")
        
        # Training dei modelli
        X_test, y_test, y_pred = estimator.train_models(X, y)
        
        # Visualizza i risultati
        estimator.plot_results(y_test, y_pred)
        
        # Salva il modello
        estimator.save_model('timeseries_diameter_estimator.pkl')
        
        # Statistiche finali
        print(f"\n=== STATISTICHE FINALI ===")
        print(f"Configurazione finestre: {window_duration}s con overlap {overlap}")
        print(f"Miglior modello: {estimator.best_model_name}")
        print(f"R² finale: {r2_score(y_test, y_pred):.4f}")
        print(f"MAE finale: {mean_absolute_error(y_test, y_pred):.4f} mm")
        print(f"RMSE finale: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} mm")
        
        # Analisi per diametro
        for diameter in np.unique(y_test):
            mask = y_test == diameter
            if mask.sum() > 0:
                mae_diameter = mean_absolute_error(y_test[mask], y_pred[mask])
                r2_diameter = r2_score(y_test[mask], y_pred[mask])
                print(f"Diametro {diameter}mm: MAE = {mae_diameter:.4f}mm, R² = {r2_diameter:.4f}")
        
        print(f"\nModello salvato come 'timeseries_diameter_estimator.pkl'")
        print("Puoi ora usare 'test_timeseries_diameter_estimator.py' per testare su nuovi dati.")
        
    except Exception as e:
        print(f"Errore durante il training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
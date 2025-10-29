"""
Script di training per il sistema ML di stima diametro equivalente cavi.
Usa le funzioni condivise da common.py

Uso:
    python train.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importa il modulo condiviso
import common as utils

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Per salvare il modello
import joblib
import json
from datetime import datetime


class CableDiameterMLPipeline:
    """
    Pipeline completa per il training e la valutazione di modelli ML
    per la stima del diametro equivalente di cavi.
    """
    
    def __init__(self, data_folder='data', train_ratio=0.7, random_state=42, feature_selection=True,
                 window_size=1.0, window_overlap=0.5):
        """
        Inizializza la pipeline.
        
        Args:
            data_folder: percorso della cartella contenente i file CSV
            train_ratio: proporzione del dataset per il training (default 0.7 = 70%)
            random_state: seed per riproducibilità
            feature_selection: se True, esegue l'analisi e selezione delle features
            window_size: durata della finestra temporale in secondi (default 1.0s)
            window_overlap: overlap tra finestre consecutive (0.5 = 50% overlap)
        """
        self.data_folder = data_folder
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.feature_selection = feature_selection
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.selected_features = None
        self.feature_analysis = {}
        
    def load_and_validate_data(self):
        """
        Carica i dati dai file CSV, valida e pulisce i dati non validi.
        
        Returns:
            DataFrame con i dati grezzi validati
        """
        print("=" * 60)
        print("FASE 1: CARICAMENTO E VALIDAZIONE DATI")
        print("=" * 60)
        
        all_data = []
        files_loaded = 0
        files_skipped = 0
        
        # Cerca tutti i file CSV nella cartella data
        data_path = Path(self.data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"La cartella '{self.data_folder}' non esiste!")
        
        csv_files = list(data_path.glob("D*_*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"Nessun file CSV trovato in '{self.data_folder}'!")
        
        print(f"Trovati {len(csv_files)} file CSV da processare\n")
        
        for file_path in csv_files:
            filename = file_path.name
            
            # Estrai il diametro equivalente dal nome del file (DXX_Y.csv)
            try:
                # Formato: DXX_Y.csv dove XX è il diametro
                parts = filename.replace('.csv', '').split('_')
                diameter_str = parts[0][1:]  # Rimuove la 'D' iniziale
                equivalent_diameter = float(diameter_str)
            except:
                print(f"⚠ SKIP: Nome file non valido: {filename}")
                files_skipped += 1
                continue
            
            # Carica il file CSV
            try:
                df = pd.read_csv(file_path)
                
                # Usa la funzione condivisa per validare e pulire
                df_clean, removed = utils.validate_and_clean_data(df, filename)
                
                # Aggiungi il diametro equivalente come colonna
                df_clean['DiametroEquivalente'] = equivalent_diameter
                df_clean['FileName'] = filename
                
                all_data.append(df_clean)
                files_loaded += 1
                
                print(f"✓ {filename}: {len(df_clean)} misure valide (rimosse: {removed})")
                
            except Exception as e:
                print(f"⚠ ERRORE nel file {filename}: {str(e)}")
                files_skipped += 1
                continue
        
        if len(all_data) == 0:
            raise ValueError("Nessun dato valido trovato nei file!")
        
        # Combina tutti i dati
        raw_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\n{'─' * 60}")
        print(f"File caricati con successo: {files_loaded}")
        print(f"File saltati: {files_skipped}")
        print(f"Totale misure valide: {len(raw_data)}")
        print(f"Diametri unici nel dataset: {raw_data['DiametroEquivalente'].nunique()}")
        print(f"{'─' * 60}\n")
        
        return raw_data
    
    def extract_features(self, raw_data):
        """
        Estrae features statistiche aggregate usando sliding windows su ogni gruppo di misure.
        Usa le funzioni condivise da cable_ml_utils.
        
        Args:
            raw_data: DataFrame con i dati grezzi
            
        Returns:
            DataFrame con features e label
        """
        print("=" * 60)
        print("FASE 2: ESTRAZIONE FEATURES CON SLIDING WINDOW")
        print("=" * 60)
        print(f"Window size: {self.window_size}s")
        print(f"Window overlap: {self.window_overlap*100:.0f}%")
        
        features_list = []
        total_windows = 0
        
        # Raggruppa per file (ogni file è una misura completa)
        for filename, group in raw_data.groupby('FileName'):
            # Estrai le serie temporali
            dx_series = group['Dx'].values
            dy_series = group['Dy'].values
            diameter = group['DiametroEquivalente'].iloc[0]
            
            # Usa la funzione condivisa per creare sliding windows ed estrarre features
            windows_features = utils.create_sliding_windows(
                dx_series, dy_series,
                window_size_seconds=self.window_size,
                window_overlap=self.window_overlap,
                estimated_duration=10.0
            )
            
            n_windows = len(windows_features)
            
            # Aggiungi metadata a ogni finestra
            for i, features in enumerate(windows_features):
                features['DiametroEquivalente'] = diameter
                features['FileName'] = filename
                features['WindowIndex'] = i
                features_list.append(features)
            
            total_windows += n_windows
            print(f"✓ {filename}: {n_windows} finestre estratte")
        
        features_df = pd.DataFrame(features_list)
        
        print(f"\n{'─' * 60}")
        print(f"Totale file processati: {raw_data['FileName'].nunique()}")
        print(f"Totale finestre create: {total_windows}")
        print(f"Media finestre per file: {total_windows / raw_data['FileName'].nunique():.1f}")
        
        # Ottieni i nomi delle features dal modulo condiviso
        feature_names = utils.get_feature_names()
        print(f"\nFeatures estratte per ogni finestra:")
        for i, feat in enumerate(feature_names, 1):
            print(f"  {i:2d}. {feat}")
        
        print(f"\nTotale features: {len(feature_names)}")
        print(f"Totale campioni (finestre): {len(features_df)}")
        print(f"{'─' * 60}\n")
        
        return features_df
    
    def analyze_and_select_features(self, features_df):
        """
        Analizza le features usando PCA e Feature Importance per selezionare quelle più rilevanti.
        
        Args:
            features_df: DataFrame con tutte le features estratte
            
        Returns:
            DataFrame con solo le features selezionate, lista nomi features selezionate
        """
        print("=" * 60)
        print("FASE 2B: ANALISI E SELEZIONE FEATURES")
        print("=" * 60)
        
        # Separa features e target
        X = features_df.drop(['DiametroEquivalente', 'FileName', 'WindowIndex'], axis=1)
        y = features_df['DiametroEquivalente']
        feature_names = X.columns.tolist()
        
        # Normalizza per l'analisi
        X_scaled = StandardScaler().fit_transform(X)
        
        print("\n1. ANALISI CORRELAZIONE CON TARGET")
        print("─" * 60)
        
        # Calcola correlazione di ogni feature con il target
        correlations = {}
        for i, col in enumerate(feature_names):
            corr = np.corrcoef(X_scaled[:, i], y)[0, 1]
            correlations[col] = abs(corr)  # Valore assoluto per considerare correlazioni negative
        
        # Ordina per correlazione decrescente
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 features per correlazione con diametro equivalente:")
        for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
            print(f"  {i:2d}. {feat:30s} | Corr: {corr:.4f}")
        
        # Identifica features con correlazione molto bassa (< 0.1)
        low_corr_features = [feat for feat, corr in sorted_corr if corr < 0.1]
        if low_corr_features:
            print(f"\n⚠ Features con correlazione < 0.1 (potenzialmente poco informative):")
            for feat in low_corr_features:
                print(f"  - {feat}: {correlations[feat]:.4f}")
        
        print("\n\n2. ANALISI PCA (Principal Component Analysis)")
        print("─" * 60)
        
        # Esegui PCA
        pca = PCA(random_state=self.random_state)
        pca.fit(X_scaled)
        
        # Varianza spiegata
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Trova quante componenti servono per spiegare il 95% della varianza
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"\nVarianza spiegata dalle prime componenti principali:")
        for i in range(min(5, len(pca.explained_variance_ratio_))):
            print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}% "
                  f"(cumulativa: {cumulative_variance[i]*100:.2f}%)")
        
        print(f"\nNumero di componenti per spiegare 95% varianza: {n_components_95}/{len(feature_names)}")
        
        # Analizza il contributo di ogni feature alle prime componenti
        print(f"\n\n3. FEATURE IMPORTANCE (Random Forest)")
        print("─" * 60)
        
        # Usa Random Forest per calcolare feature importance
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf_temp.fit(X_scaled, y)
        
        importances = dict(zip(feature_names, rf_temp.feature_importances_))
        sorted_importance = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 features per importanza (Random Forest):")
        for i, (feat, imp) in enumerate(sorted_importance[:15], 1):
            print(f"  {i:2d}. {feat:30s} | Importance: {imp:.4f}")
        
        # Identifica features con importanza molto bassa
        threshold_importance = 0.01
        low_imp_features = [feat for feat, imp in sorted_importance if imp < threshold_importance]
        if low_imp_features:
            print(f"\n⚠ Features con importanza < {threshold_importance} (potenzialmente ridondanti):")
            for feat in low_imp_features:
                print(f"  - {feat}: {importances[feat]:.4f}")
        
        print("\n\n4. ANALISI MULTICOLLINEARITÀ")
        print("─" * 60)
        
        # Calcola matrice di correlazione tra features
        corr_matrix = np.corrcoef(X_scaled.T)
        
        # Trova coppie di features altamente correlate (possibile ridondanza)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.9:  # Soglia alta correlazione
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
        
        if high_corr_pairs:
            print(f"\nCoppie di features altamente correlate (>0.9) - possibile ridondanza:")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Mostra max 10
                print(f"  - {feat1} <-> {feat2}: {corr:.4f}")
        else:
            print("\n✓ Nessuna coppia di features con correlazione >0.9")
        
        print("\n\n5. STRATEGIA DI SELEZIONE")
        print("─" * 60)
        
        # Strategia: combina i tre criteri
        # 1. Feature Importance > threshold
        # 2. Correlazione con target > 0.05
        # 3. Rimuovi una delle due features se altamente correlate tra loro
        
        # Crea uno score combinato
        feature_scores = {}
        for feat in feature_names:
            # Normalizza importance e correlazione tra 0 e 1
            imp_norm = importances[feat] / max(importances.values())
            corr_norm = correlations[feat] / max(correlations.values()) if max(correlations.values()) > 0 else 0
            
            # Score combinato (puoi pesare diversamente)
            feature_scores[feat] = (imp_norm * 0.6 + corr_norm * 0.4)  # 60% importance, 40% correlazione
        
        # Ordina per score
        sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Selezione: prendi top features, ma rimuovi quelle ridondanti
        selected = []
        selected_corr_vectors = []
        
        for feat, score in sorted_scores:
            # Controlla se questa feature è troppo correlata con una già selezionata
            feat_idx = feature_names.index(feat)
            feat_vector = X_scaled[:, feat_idx]
            
            is_redundant = False
            for selected_vector in selected_corr_vectors:
                if abs(np.corrcoef(feat_vector, selected_vector)[0, 1]) > 0.85:
                    is_redundant = True
                    break
            
            if not is_redundant and score > 0.1:  # Soglia minima di score
                selected.append(feat)
                selected_corr_vectors.append(feat_vector)
        
        # Assicurati di avere almeno 8 features
        if len(selected) < 8:
            selected = [feat for feat, _ in sorted_scores[:8]]
        
        print(f"\n✓ Features selezionate: {len(selected)}/{len(feature_names)}")
        print("\nFeatures finali scelte:")
        for i, feat in enumerate(selected, 1):
            imp = importances[feat]
            corr = correlations[feat]
            print(f"  {i:2d}. {feat:30s} | Importance: {imp:.4f} | Corr: {corr:.4f}")
        
        # Features rimosse
        removed = [f for f in feature_names if f not in selected]
        if removed:
            print(f"\nFeatures rimosse ({len(removed)}):")
            for feat in removed:
                reason = []
                if importances[feat] < 0.01:
                    reason.append("bassa importance")
                if correlations[feat] < 0.05:
                    reason.append("bassa correlazione")
                if any(abs(corr_matrix[feature_names.index(feat), feature_names.index(s)]) > 0.85 for s in selected if s != feat):
                    reason.append("ridondante")
                print(f"  - {feat:30s} ({', '.join(reason) if reason else 'score basso'})")
        
        print(f"\n{'─' * 60}\n")
        
        # Salva l'analisi
        self.feature_analysis = {
            'all_features': feature_names,
            'selected_features': selected,
            'removed_features': removed,
            'correlations': correlations,
            'importances': importances,
            'pca_variance_explained': pca.explained_variance_ratio_.tolist(),
            'n_components_95': int(n_components_95)
        }
        
        self.selected_features = selected
        
        # Restituisci il DataFrame con solo le features selezionate
        features_selected_df = features_df[selected + ['DiametroEquivalente', 'FileName', 'WindowIndex']].copy()
        
        return features_selected_df, selected
    
    def prepare_train_test_split(self, features_df, selected_features=None):
        """
        Prepara i set di training e test.
        IMPORTANTE: Lo split avviene a livello di FILE, non di finestre,
        per evitare data leakage (finestre dello stesso file vanno tutte insieme).
        
        Args:
            features_df: DataFrame con features estratte
            selected_features: lista di features da usare (se None, usa tutte)
            
        Returns:
            X_train, X_test, y_train, y_test (tutti scalati)
        """
        print("=" * 60)
        print("FASE 3: PREPARAZIONE TRAIN/TEST SET")
        print("=" * 60)
        print("⚠ SPLIT A LIVELLO DI FILE per evitare data leakage")
        
        # Ottieni la lista di file unici
        unique_files = features_df['FileName'].unique()
        n_files = len(unique_files)
        
        # Dividi i FILE in train e test
        np.random.seed(self.random_state)
        shuffled_files = np.random.permutation(unique_files)
        n_train_files = int(n_files * self.train_ratio)
        
        train_files = shuffled_files[:n_train_files]
        test_files = shuffled_files[n_train_files:]
        
        # Crea i set basandosi sui file
        train_mask = features_df['FileName'].isin(train_files)
        test_mask = features_df['FileName'].isin(test_files)
        
        train_df = features_df[train_mask].copy()
        test_df = features_df[test_mask].copy()
        
        # Separa features e label
        cols_to_drop = ['DiametroEquivalente', 'FileName', 'WindowIndex']
        
        if selected_features is not None:
            X_train = train_df[selected_features]
            X_test = test_df[selected_features]
            print(f"Usando {len(selected_features)} features selezionate")
        else:
            X_train = train_df.drop(cols_to_drop, axis=1)
            X_test = test_df.drop(cols_to_drop, axis=1)
            print(f"Usando tutte le {len(X_train.columns)} features")
        
        y_train = train_df['DiametroEquivalente']
        y_test = test_df['DiametroEquivalente']
        
        # Normalizzazione delle features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nSplit a livello FILE:")
        print(f"  Train files: {n_train_files} file ({self.train_ratio*100:.0f}%)")
        print(f"  Test files:  {len(test_files)} file ({(1-self.train_ratio)*100:.0f}%)")
        
        print(f"\nRisultato in finestre:")
        print(f"  Train set: {len(X_train)} finestre")
        print(f"  Test set:  {len(X_test)} finestre")
        print(f"  Features utilizzate: {X_train.shape[1]}")
        
        print(f"\nDistribuzione diametri nel train set:")
        print(y_train.value_counts().sort_index())
        print(f"\nDistribuzione diametri nel test set:")
        print(y_test.value_counts().sort_index())
        print(f"{'─' * 60}\n")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """
        Inizializza i 5 modelli di ML da testare.
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True
            )
        }
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Addestra e valuta tutti i modelli.
        
        Args:
            X_train, X_test, y_train, y_test: set di training e test
        """
        print("=" * 60)
        print("FASE 4: TRAINING E VALUTAZIONE MODELLI")
        print("=" * 60)
        
        self.initialize_models()
        
        for model_name, model in self.models.items():
            print(f"\n{'─' * 60}")
            print(f"Modello: {model_name}")
            print(f"{'─' * 60}")
            
            # Training
            print("Training in corso...", end=" ")
            model.fit(X_train, y_train)
            print("✓")
            
            # Predizioni
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calcola metriche
            train_metrics = self.calculate_metrics(y_train, y_train_pred, "Train")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")
            
            # Salva risultati
            self.results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'y_test_true': y_test,
                'y_test_pred': y_test_pred
            }
            
            # Stampa risultati
            print(f"\nRISULTATI {model_name}:")
            print(f"  TRAIN SET:")
            self.print_metrics(train_metrics, indent=4)
            print(f"  TEST SET:")
            self.print_metrics(test_metrics, indent=4)
        
        print(f"\n{'=' * 60}\n")
    
    def calculate_metrics(self, y_true, y_pred, dataset_name=""):
        """
        Calcola le metriche di valutazione per la regressione.
        
        Metriche calcolate:
        - MAE (Mean Absolute Error): errore medio assoluto
        - RMSE (Root Mean Squared Error): radice dell'errore quadratico medio
        - R² (R-squared): coefficiente di determinazione (0-1, meglio se vicino a 1)
        - MAPE (Mean Absolute Percentage Error): errore percentuale medio
        
        Args:
            y_true: valori veri
            y_pred: valori predetti
            dataset_name: nome del dataset (per logging)
            
        Returns:
            dict con le metriche
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE con gestione divisione per zero
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def print_metrics(self, metrics, indent=0):
        """Stampa le metriche in formato leggibile."""
        prefix = " " * indent
        print(f"{prefix}MAE:  {metrics['MAE']:.4f} mm")
        print(f"{prefix}RMSE: {metrics['RMSE']:.4f} mm")
        print(f"{prefix}R²:   {metrics['R2']:.4f}")
        print(f"{prefix}MAPE: {metrics['MAPE']:.2f}%")
    
    def select_best_model(self):
        """
        Seleziona il miglior modello basandosi su R² sul test set.
        """
        print("=" * 60)
        print("FASE 5: SELEZIONE MIGLIOR MODELLO")
        print("=" * 60)
        
        best_r2 = -np.inf
        best_name = None
        
        print("\nCLASSIFICA MODELLI (ordinati per R² su test set):\n")
        
        # Ordina i modelli per R² decrescente
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_metrics']['R2'],
            reverse=True
        )
        
        for rank, (model_name, results) in enumerate(sorted_models, 1):
            r2 = results['test_metrics']['R2']
            mae = results['test_metrics']['MAE']
            
            marker = "🏆" if rank == 1 else f"{rank}."
            print(f"{marker} {model_name:20s} - R²: {r2:.4f} | MAE: {mae:.4f} mm")
            
            if r2 > best_r2:
                best_r2 = r2
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        print(f"\n{'─' * 60}")
        print(f"🏆 MIGLIOR MODELLO: {self.best_model_name}")
        print(f"{'─' * 60}")
        print(f"Metriche sul test set:")
        self.print_metrics(self.results[best_name]['test_metrics'], indent=2)
        print(f"{'─' * 60}\n")
        
        return self.best_model_name, self.best_model
    
    def save_best_model(self, output_folder='models'):
        """
        Salva il miglior modello e i relativi metadati.
        
        Args:
            output_folder: cartella dove salvare il modello
        """
        print("=" * 60)
        print("FASE 6: SALVATAGGIO MODELLO")
        print("=" * 60)
        
        # Crea la cartella se non esiste
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_model_{timestamp}.joblib"
        scaler_filename = f"scaler_{timestamp}.joblib"
        metadata_filename = f"metadata_{timestamp}.json"
        
        # Salva il modello
        model_path = output_path / model_filename
        joblib.dump(self.best_model, model_path)
        print(f"✓ Modello salvato: {model_path}")
        
        # Salva lo scaler
        scaler_path = output_path / scaler_filename
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler salvato: {scaler_path}")
        
        # Prepara e salva i metadati
        metadata = {
            'timestamp': timestamp,
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'train_ratio': self.train_ratio,
            'random_state': self.random_state,
            'window_size': self.window_size,
            'window_overlap': self.window_overlap,
            'test_metrics': {
                k: float(v) if not np.isinf(v) else 'inf'
                for k, v in self.results[self.best_model_name]['test_metrics'].items()
            },
            'train_metrics': {
                k: float(v) if not np.isinf(v) else 'inf'
                for k, v in self.results[self.best_model_name]['train_metrics'].items()
            },
            'all_models_comparison': {
                name: {
                    'test_R2': float(res['test_metrics']['R2']),
                    'test_MAE': float(res['test_metrics']['MAE'])
                }
                for name, res in self.results.items()
            },
            'feature_analysis': self.feature_analysis if self.feature_selection else None,
            'selected_features': self.selected_features if self.feature_selection else None
        }
        
        metadata_path = output_path / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata salvati: {metadata_path}")
        
        print(f"{'─' * 60}\n")
        
        return str(model_path), str(scaler_path), str(metadata_path)
    
    def run_complete_pipeline(self):
        """
        Esegue l'intera pipeline dall'inizio alla fine.
        
        Returns:
            tuple: (best_model_name, best_model, model_path)
        """
        print("\n" + "=" * 60)
        print("PIPELINE ML - STIMA DIAMETRO EQUIVALENTE CAVI")
        print("=" * 60 + "\n")
        
        # 1. Carica e valida i dati
        raw_data = self.load_and_validate_data()
        
        # 2. Estrai features
        features_df = self.extract_features(raw_data)
        
        # 2B. Analizza e seleziona features (se abilitato)
        if self.feature_selection:
            features_df, selected_features = self.analyze_and_select_features(features_df)
        else:
            selected_features = None
        
        # 3. Prepara train/test split
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(features_df, selected_features)
        
        # 4. Addestra e valuta i modelli
        self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # 5. Seleziona il miglior modello
        best_model_name, best_model = self.select_best_model()
        
        # 6. Salva il miglior modello
        model_path, scaler_path, metadata_path = self.save_best_model()
        
        print("=" * 60)
        print("PIPELINE COMPLETATA CON SUCCESSO! ✓")
        print("=" * 60)
        print(f"\nFile salvati:")
        print(f"  - Modello:  {model_path}")
        print(f"  - Scaler:   {scaler_path}")
        print(f"  - Metadata: {metadata_path}")
        print("\n" + "=" * 60 + "\n")
        
        return best_model_name, best_model, model_path


def main():
    """
    Funzione principale per eseguire la pipeline.
    """
    # Parametri configurabili
    DATA_FOLDER = 'data'
    TRAIN_RATIO = 0.7  # 70% training, 30% test
    RANDOM_STATE = 42
    FEATURE_SELECTION = True  # Abilita l'analisi e selezione features
    WINDOW_SIZE = 1.0  # Durata finestra in secondi
    WINDOW_OVERLAP = 0.5  # Overlap tra finestre (50%)
    
    # Crea ed esegui la pipeline
    pipeline = CableDiameterMLPipeline(
        data_folder=DATA_FOLDER,
        train_ratio=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        feature_selection=FEATURE_SELECTION,
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP
    )
    
    # Esegui la pipeline completa
    best_model_name, best_model, model_path = pipeline.run_complete_pipeline()
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
# src/forecast.py
"""
Module de prévision pour IA_Logistique_Durable
Modèles : XGBoost et LSTM
Prévisions : 1 jour et 7 jours (au lieu de 1 mois et 12 mois)
"""

import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os


class Forecaster:
    def __init__(self, db_table: str = 'commandes'):
        self.db_table = db_table
        self.data = DataPreprocessor.preprocess_pipeline(table_name=db_table)

        # ============ LIGNES DE DEBUG ============
        print("\n📊 DEBUG - Colonnes disponibles :")
        print(self.data.columns.tolist())
        print(f"\n📏 Nombre de lignes : {len(self.data)}")
        print(f"\n📐 Dimensions du DataFrame : {self.data.shape}")
        print("\n🔍 Aperçu des 5 premières lignes :")
        print(self.data.head())
        print("\n" + "=" * 60 + "\n")
        # =========================================

        # Créer le répertoire models s'il n'existe pas
        os.makedirs('models', exist_ok=True)

    # -----------------------------
    # PRÉPARATION DES DONNÉES POUR LA SÉRIE TEMPORELLE
    # -----------------------------
    def prepare_series(self, target_column: str, period: str = 'D'):
        """
        Agrège les données par jour, semaine ou mois
        period: 'D' = journalier, 'W' = hebdomadaire, 'ME' = mensuel
        """
        df = self.data.copy()
        df['Date_commande'] = pd.to_datetime(df['Date_commande'])

        # On garde seulement la colonne numérique pour le sum()
        df_grouped = df.groupby(pd.Grouper(key='Date_commande', freq=period))[target_column].sum().reset_index()

        # Supprimer les lignes avec des valeurs à zéro
        df_grouped = df_grouped[df_grouped[target_column] > 0]

        return df_grouped[['Date_commande', target_column]]

    # -----------------------------
    # XGBoost
    # -----------------------------
    def train_xgboost(self, target_column: str, horizon: int = 1):
        df = self.prepare_series(target_column, period='D')  # Journalier au lieu de mensuel

        print(f"\n🔍 Données après agrégation journalière :")
        print(f"Nombre de jours : {len(df)}")
        print(df)

        if len(df) < 2:
            print("\n⚠️ Pas assez de données pour entraîner XGBoost (minimum 2 jours nécessaires)")
            return None

        df['lag_1'] = df[target_column].shift(1)
        df = df.dropna()

        if len(df) == 0:
            print("\n⚠️ Aucune donnée après création du lag")
            return None

        X = df[['lag_1']]
        y = df[target_column]

        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)

        # Prédiction du prochain horizon
        last_lag = np.array([[df[target_column].iloc[-1]]])
        preds = []
        for _ in range(horizon):
            pred = model.predict(last_lag)[0]
            preds.append(pred)
            last_lag = np.array([[pred]])

        # Sauvegarde du modèle
        joblib.dump(model, f"models/xgb_{target_column}.pkl")
        return preds

    # -----------------------------
    # LSTM
    # -----------------------------
    def train_lstm(self, target_column: str, horizon: int = 1, epochs: int = 50):
        df = self.prepare_series(target_column, period='D')  # Journalier au lieu de mensuel

        print(f"\n🔍 Données après agrégation journalière :")
        print(f"Nombre de jours : {len(df)}")
        print(df)

        if len(df) < 2:
            print("\n⚠️ Pas assez de données pour entraîner LSTM (minimum 2 jours nécessaires)")
            return None

        series = df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)

        # Création des séquences
        X, y = [], []
        for i in range(1, len(series_scaled)):
            X.append(series_scaled[i - 1:i, 0])
            y.append(series_scaled[i, 0])
        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            print("\n⚠️ Pas assez de données pour créer des séquences")
            return None

        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='loss', patience=5)
        model.fit(X, y, epochs=epochs, batch_size=1, verbose=0, callbacks=[early_stop])

        # Prédiction horizon
        input_seq = series_scaled[-1].reshape(1, 1, 1)
        preds_scaled = []
        for _ in range(horizon):
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            preds_scaled.append(pred_scaled)
            input_seq = np.array(pred_scaled).reshape(1, 1, 1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        # Sauvegarde
        model.save(f"models/lstm_{target_column}.h5")
        return preds


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer une instance du forecaster
    forecaster = Forecaster(db_table='commandes')

    # Prévision avec XGBoost pour 1 jour
    print("\n" + "=" * 60)
    print("=== Prévision XGBoost (1 jour) ===")
    print("=" * 60)
    xgb_preds_1d = forecaster.train_xgboost(target_column='Quantite_commandee', horizon=1)
    if xgb_preds_1d is not None:
        print(f"✅ Prévision 1 jour: {xgb_preds_1d}")

    # Prévision avec XGBoost pour 7 jours
    print("\n" + "=" * 60)
    print("=== Prévision XGBoost (7 jours) ===")
    print("=" * 60)
    xgb_preds_7d = forecaster.train_xgboost(target_column='Quantite_commandee', horizon=7)
    if xgb_preds_7d is not None:
        print(f"✅ Prévisions 7 jours: {xgb_preds_7d}")

    # Prévision avec LSTM pour 1 jour
    print("\n" + "=" * 60)
    print("=== Prévision LSTM (1 jour) ===")
    print("=" * 60)
    lstm_preds_1d = forecaster.train_lstm(target_column='Quantite_commandee', horizon=1, epochs=50)
    if lstm_preds_1d is not None:
        print(f"✅ Prévision 1 jour: {lstm_preds_1d}")

    # Prévision avec LSTM pour 7 jours
    print("\n" + "=" * 60)
    print("=== Prévision LSTM (7 jours) ===")
    print("=" * 60)
    lstm_preds_7d = forecaster.train_lstm(target_column='Quantite_commandee', horizon=7, epochs=50)
    if lstm_preds_7d is not None:
        print(f"✅ Prévisions 7 jours: {lstm_preds_7d}")
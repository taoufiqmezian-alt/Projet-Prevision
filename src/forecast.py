"""
Module de prévision pour IA_Logistique_Durable
Modèles : XGBoost et LSTM
Prévisions : 30 jours et 12 mois
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from src.preprocessing import DataPreprocessor

# Supprimer warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


class Forecaster:
    def __init__(self, db_table: str = 'commandes'):
        self.db_table = db_table
        all_data = DataPreprocessor.load_data_from_db()
        if db_table not in all_data:
            raise ValueError(f"La table '{db_table}' est introuvable dans la base SQLite.")
        df = all_data[db_table]
        self.data = DataPreprocessor.preprocess_table(df, table_name=db_table)
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        print(f"\n✅ Forecaster initialisé avec {len(self.data)} lignes de données.")

    def prepare_series(self, target_column: str, period: str = 'D'):
        df = self.data.copy()
        df['Date_commande'] = pd.to_datetime(df['Date_commande'])
        df_grouped = df.groupby(pd.Grouper(key='Date_commande', freq=period))[target_column].sum().reset_index()
        df_grouped = df_grouped[df_grouped[target_column] > 0]
        return df_grouped

    def train_xgboost(self, target_column: str, horizon: int = 1, period: str = 'D'):
        df = self.prepare_series(target_column, period=period)
        if len(df) < 2:
            print("⚠️ Pas assez de données pour XGBoost.")
            return None, df
        df['lag_1'] = df[target_column].shift(1)
        df = df.dropna()
        X, y = df[['lag_1']], df[target_column]
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        last_lag = np.array([[df[target_column].iloc[-1]]])
        preds = []
        for _ in range(horizon):
            pred = model.predict(last_lag)[0]
            preds.append(float(pred))
            last_lag = np.array([[pred]])
        joblib.dump(model, f"models/xgb_{target_column}_{period}.pkl")
        return preds, df

    def train_lstm(self, target_column: str, horizon: int = 1, period: str = 'D', epochs: int = 50):
        df = self.prepare_series(target_column, period=period)
        if len(df) < 2:
            print("⚠️ Pas assez de données pour LSTM.")
            return None, df
        series = df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series)
        X, y = [], []
        for i in range(1, len(series_scaled)):
            X.append(series_scaled[i - 1:i, 0])
            y.append(series_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([Input(shape=(X.shape[1], 1)), LSTM(50, activation='relu'), Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=epochs, batch_size=1, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5)])
        input_seq = series_scaled[-1].reshape(1, 1, 1)
        preds_scaled = []
        for _ in range(horizon):
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            preds_scaled.append(pred_scaled)
            input_seq = np.array(pred_scaled).reshape(1, 1, 1)
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        model.save(f"models/lstm_{target_column}_{period}.keras")
        return preds, df

    def plot_forecast(self, df, preds, horizon, title, period):
        last_date = df['Date_commande'].iloc[-1]
        if period == 'D':
            future_dates = pd.date_range(last_date, periods=horizon+1, freq='D')[1:]
        elif period == 'ME':
            future_dates = pd.date_range(last_date, periods=horizon+1, freq='M')[1:]
        forecast_df = pd.DataFrame({'Date_commande': future_dates, 'Prévision': preds})
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date_commande'], df['Quantite_commandee'], label='Historique', marker='o')
        plt.plot(forecast_df['Date_commande'], forecast_df['Prévision'], label='Prévision', marker='x', color='red')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Quantité commandée")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/{title.replace(' ', '_')}.png")
        plt.close()
        # Affichage console
        print(f"\n📊 {title}")
        print(f"{'Date':<15} | {'Prévision':>10}")
        print("-"*30)
        for d, p in zip(forecast_df['Date_commande'], forecast_df['Prévision']):
            print(f"{pd.to_datetime(d).strftime('%Y-%m-%d'):<15} | {p:>10.2f}")

    def print_summary(self, preds, horizon, title):
        total = sum(preds)
        moyenne = total / horizon
        print(f"\n📌 Résumé {title} :")
        print(f"   Total prévision : {total:.2f} unités")
        print(f"   Moyenne par période : {moyenne:.2f} unités\n")


# ========================
# EXÉCUTION PRINCIPALE
# ========================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 SYSTÈME DE PRÉVISION - IA LOGISTIQUE DURABLE")
    print("="*80)

    forecaster = Forecaster(db_table='commandes')
    target = 'Quantite_commandee'

    # 30 jours
    xgb_preds_30, df_xgb_30 = forecaster.train_xgboost(target, horizon=30, period='D')
    forecaster.plot_forecast(df_xgb_30, xgb_preds_30, 30, "Prévision_XGBoost_30_Jours", 'D')
    forecaster.print_summary(xgb_preds_30, 30, "XGBoost 30 Jours")

    lstm_preds_30, df_lstm_30 = forecaster.train_lstm(target, horizon=30, period='D', epochs=50)
    forecaster.plot_forecast(df_lstm_30, lstm_preds_30, 30, "Prévision_LSTM_30_Jours", 'D')
    forecaster.print_summary(lstm_preds_30, 30, "LSTM 30 Jours")

    # 12 mois
    xgb_preds_12m, df_xgb_12m = forecaster.train_xgboost(target, horizon=12, period='ME')
    forecaster.plot_forecast(df_xgb_12m, xgb_preds_12m, 12, "Prévision_XGBoost_12_Mois", 'ME')
    forecaster.print_summary(xgb_preds_12m, 12, "XGBoost 12 Mois")

    lstm_preds_12m, df_lstm_12m = forecaster.train_lstm(target, horizon=12, period='ME', epochs=50)
    forecaster.plot_forecast(df_lstm_12m, lstm_preds_12m, 12, "Prévision_LSTM_12_Mois", 'ME')
    forecaster.print_summary(lstm_preds_12m, 12, "LSTM 12 Mois")

    print("\n✅ Prévisions terminées — résultats dans le dossier 'outputs/'\n")

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
import streamlit as st
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ==============================
# CONFIGURATION
# ==============================
DB_PATH = "data/prevision.db"
st.set_page_config(page_title="Prévision Logistique ML/DL", layout="wide")

# ==============================
# UTILITAIRES
# ==============================
@st.cache_data
def load_data(table):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

@st.cache_data
def get_product_price_dict():
    df = load_data("produits")
    return dict(zip(df["ID_produit"], df["Cout_par_unite"]))

def creer_features(df):
    df = df.copy()
    df["Date_commande"] = pd.to_datetime(df["Date_commande"])
    df["Jours"] = (df["Date_commande"] - df["Date_commande"].min()).dt.days
    df["mois"] = df["Date_commande"].dt.month
    df["jour_semaine"] = df["Date_commande"].dt.dayofweek
    df["sin_mois"] = np.sin(2*np.pi*df["mois"]/12)
    df["cos_mois"] = np.cos(2*np.pi*df["mois"]/12)
    return df

def calculate_ca(df, price_dict):
    df["Chiffre_affaire"] = df["Prévision"] * df["ID_produit"].map(price_dict)
    return df

# ==============================
# ENTRAÎNEMENT XGBOOST (1 FOIS)
# ==============================
@st.cache_resource
def train_xgb_models(df_history):
    models = {}
    for prod in df_history["ID_produit"].unique():
        df = df_history[df_history["ID_produit"] == prod]
        df = creer_features(df)

        X = df[["Jours","mois","jour_semaine","sin_mois","cos_mois"]]
        y = df["Quantite"]

        model = XGBRegressor(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42
        )
        model.fit(X, y)
        models[prod] = model
    return models

# ==============================
# ENTRAÎNEMENT LSTM (1 FOIS)
# ==============================
@st.cache_resource
def train_lstm_models(df_history):
    models = {}
    scalers = {}

    for prod in df_history["ID_produit"].unique():
        df = df_history[df_history["ID_produit"] == prod]
        df = creer_features(df)

        features = ["Jours","sin_mois","cos_mois","jour_semaine"]
        X = df[features].values
        y = df["Quantite"].values.reshape(-1,1)

        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xs, ys = sx.fit_transform(X), sy.fit_transform(y)
        Xs = Xs.reshape((Xs.shape[0],1,Xs.shape[1]))

        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(1,Xs.shape[2])),
            Dropout(0.2),
            LSTM(16),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        model.fit(
            Xs, ys,
            epochs=80,
            batch_size=16,
            verbose=0,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )

        models[prod] = model
        scalers[prod] = (sx, sy)

    return models, scalers

# ==============================
# PRÉVISION RAPIDE (XGB)
# ==============================
def forecast_xgb_fast(df_history, models, horizon):
    forecasts = []
    price_dict = get_product_price_dict()

    for prod, model in models.items():
        df = df_history[df_history["ID_produit"] == prod]
        last_date = pd.to_datetime(df["Date_commande"].max())
        base_days = (pd.to_datetime(df["Date_commande"]) -
                     pd.to_datetime(df["Date_commande"]).min()).dt.days.max()

        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]
        df_future = pd.DataFrame({"Date_commande": future_dates})
        df_future = creer_features(df_future)
        df_future["Jours"] = range(base_days+1, base_days+horizon+1)

        Xf = df_future[["Jours","mois","jour_semaine","sin_mois","cos_mois"]]
        y_pred = np.maximum(model.predict(Xf),0)

        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "ID_produit": prod,
            "Prévision": y_pred
        }))

    df_forecast = pd.concat(forecasts, ignore_index=True)
    return calculate_ca(df_forecast, price_dict)

# ==============================
# PRÉVISION RAPIDE (LSTM)
# ==============================
def forecast_lstm_fast(df_history, models, scalers, horizon):
    forecasts = []
    price_dict = get_product_price_dict()

    for prod, model in models.items():
        sx, sy = scalers[prod]
        df = df_history[df_history["ID_produit"] == prod]

        last_date = pd.to_datetime(df["Date_commande"].max())
        base_days = (pd.to_datetime(df["Date_commande"]) -
                     pd.to_datetime(df["Date_commande"]).min()).dt.days.max()

        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon+1)]
        df_future = pd.DataFrame({"Date_commande": future_dates})
        df_future = creer_features(df_future)
        df_future["Jours"] = range(base_days+1, base_days+horizon+1)

        Xf = df_future[["Jours","sin_mois","cos_mois","jour_semaine"]].values
        Xf = sx.transform(Xf).reshape((Xf.shape[0],1,Xf.shape[1]))

        y_pred = sy.inverse_transform(model.predict(Xf, verbose=0)).flatten()
        y_pred = np.maximum(y_pred,0)

        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "ID_produit": prod,
            "Prévision": y_pred
        }))

    df_forecast = pd.concat(forecasts, ignore_index=True)
    return calculate_ca(df_forecast, price_dict)

# ==============================
# STREAMLIT APP
# ==============================
def main():
    st.title("📦 Prévision Logistique – XGBoost & LSTM")

    st.sidebar.header("⚙️ Paramètres")

    horizon = st.sidebar.number_input(
        "Durée de prévision (jours)",
        min_value=1,
        max_value=365,
        value=None,
        placeholder="Ex : 7, 30, 90"
    )

    model_choice = st.sidebar.radio(
        "Modèle",
        ["XGBoost (rapide)", "LSTM (plus précis)"]
    )

    lancer = st.sidebar.button("🚀 Lancer la prévision")

    if lancer:
        if horizon is None:
            st.sidebar.error("❌ Veuillez saisir une durée.")
            st.stop()

        df_history = load_data("historique_commandes")

        if model_choice == "XGBoost (rapide)":
            models = train_xgb_models(df_history)
            df_forecast = forecast_xgb_fast(df_history, models, horizon)
        else:
            models, scalers = train_lstm_models(df_history)
            df_forecast = forecast_lstm_fast(df_history, models, scalers, horizon)

        st.success("✅ Prévision terminée")

        st.dataframe(df_forecast, use_container_width=True)

        st.subheader("📈 Graphiques (limités)")
        for prod in df_forecast["ID_produit"].unique()[:4]:
            d = df_forecast[df_forecast["ID_produit"] == prod]
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(d["Date"], d["Prévision"], marker="o")
            ax.set_title(f"Produit {prod}")
            ax.grid(True)
            st.pyplot(fig)

if __name__ == "__main__":
    main()

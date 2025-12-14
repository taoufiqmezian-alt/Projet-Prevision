import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# --- CONFIGURATION (Identique à votre script) ---
DB_PATH = "data/prevision.db"  # Assurez-vous que ce chemin est accessible dans l'environnement Streamlit
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# FONCTIONS UTILITAIRES (Inchangées, sauf plot_forecast)
# -----------------------------
@st.cache_data  # Mettre en cache les données chargées
def load_data(table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


@st.cache_data
def get_product_price_dict() -> dict:
    df_prod = load_data("produits")
    return dict(zip(df_prod["ID_produit"], df_prod["Cout_par_unite"]))


def calculate_chiffre_affaire(df_forecast: pd.DataFrame, prix_dict: dict) -> pd.DataFrame:
    df = df_forecast.copy()
    df["Chiffre_affaire"] = df.apply(
        lambda row: row["Prévision"] * prix_dict.get(row["ID_produit"], 0), axis=1
    )
    return df


def creer_features_enrichies(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features avancées pour le Machine Learning"""
    df = df.copy()
    df['Date_commande'] = pd.to_datetime(df['Date_commande'])
    df['annee'] = df['Date_commande'].dt.year
    df['mois'] = df['Date_commande'].dt.month
    df['jour_semaine'] = df['Date_commande'].dt.dayofweek
    df['jour_annee'] = df['Date_commande'].dt.dayofyear
    df['trimestre'] = df['Date_commande'].dt.quarter
    df['semaine'] = df['Date_commande'].dt.isocalendar().week
    df['sin_jour_annee'] = np.sin(2 * np.pi * df['jour_annee'] / 365.25)
    df['cos_jour_annee'] = np.cos(2 * np.pi * df['jour_annee'] / 365.25)
    df['sin_mois'] = np.sin(2 * np.pi * df['mois'] / 12)
    df['cos_mois'] = np.cos(2 * np.pi * df['mois'] / 12)
    df['sin_semaine'] = np.sin(2 * np.pi * df['semaine'] / 52)
    df['cos_semaine'] = np.cos(2 * np.pi * df['semaine'] / 52)
    df['est_weekend'] = (df['jour_semaine'] >= 5).astype(int)
    df['debut_mois'] = (df['Date_commande'].dt.day <= 7).astype(int)
    df['fin_mois'] = (df['Date_commande'].dt.day >= 24).astype(int)
    df['haute_saison'] = ((df['mois'] >= 11) | (df['mois'] <= 1)).astype(int)
    df['Jours'] = (df['Date_commande'] - df['Date_commande'].min()).dt.days
    return df


def evaluer_modele(y_true, y_pred):
    """Calcule les métriques de performance"""
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'mape': mape, 'rmse': rmse, 'mae': mae}


def get_forecast_summary(df_forecast: pd.DataFrame) -> dict:
    """Calcule le résumé pour l'affichage Streamlit"""
    total_units = df_forecast["Prévision"].sum()
    avg_units = df_forecast["Prévision"].mean()
    total_ca = df_forecast["Chiffre_affaire"].sum()
    avg_ca = df_forecast["Chiffre_affaire"].mean()

    return {
        "Total_unites": f"{total_units:,.0f}",
        "Moyenne_unites": f"{avg_units:,.1f}",
        "Total_CA": f"{total_ca:,.2f} €",
        "Moyenne_CA": f"{avg_ca:,.2f} €"
    }


def plot_forecast_st(df_forecast: pd.DataFrame, model_name: str, horizon_name: str) -> list:
    """MODIFIÉ : Retourne une liste de figures Matplotlib pour Streamlit"""
    figures = []
    for prod_id in df_forecast["ID_produit"].unique():
        df_prod = df_forecast[df_forecast["ID_produit"] == prod_id]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_prod["Date"], df_prod["Prévision"], marker='o', label='Prévision unités')
        ax.set_title(f"{model_name} - {horizon_name} - Produit {prod_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Prévision unités")
        ax.grid(True)
        ax.legend()
        figures.append(fig)
    return figures


# -----------------------------
# PRÉVISION XGBOOST OPTIMISÉ (ML) - Encapsulé pour Streamlit
# -----------------------------
@st.cache_resource
def forecast_xgb(df_history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Contient votre logique XGBoost complète
    forecasts = []
    prix_dict = get_product_price_dict()

    for prod_id in df_history["ID_produit"].unique():
        df_prod = df_history[df_history["ID_produit"] == prod_id].copy()
        df_prod = creer_features_enrichies(df_prod)
        df_prod.sort_values("Date_commande", inplace=True)

        features = ['Jours', 'mois', 'jour_semaine', 'trimestre',
                    'sin_jour_annee', 'cos_jour_annee', 'sin_mois', 'cos_mois',
                    'est_weekend', 'debut_mois', 'fin_mois', 'haute_saison']

        X = df_prod[features].values
        y = df_prod["Quantite"].values

        # Logique d'entraînement et de prévision... (Identique à votre code)
        if len(X) > 30:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            model = XGBRegressor(
                objective="reg:squarederror", n_estimators=300, max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=42, gamma=0.1, min_child_weight=3
            )
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_test = np.maximum(y_pred_test, 0)
            # Pas d'affichage print ici, Streamlit s'en charge
        else:
            model = XGBRegressor(
                objective="reg:squareor", n_estimators=300, max_depth=6, learning_rate=0.05
            )
            model.fit(X, y)

        last_date = df_prod["Date_commande"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        df_future = pd.DataFrame({'Date_commande': future_dates})
        df_future = creer_features_enrichies(df_future)
        last_day = df_prod['Jours'].max()
        df_future['Jours'] = [last_day + i for i in range(1, horizon + 1)]

        X_future = df_future[features].values
        y_pred = model.predict(X_future)
        y_pred = np.maximum(y_pred, 0)

        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "ID_produit": prod_id,
            "Prévision": y_pred
        }))

    df_forecast = pd.concat(forecasts, ignore_index=True)
    df_forecast = calculate_chiffre_affaire(df_forecast, prix_dict)
    return df_forecast


# -----------------------------
# PRÉVISION LSTM OPTIMISÉ (DL) - Encapsulé pour Streamlit
# -----------------------------
@st.cache_resource
def forecast_lstm(df_history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Contient votre logique LSTM complète
    forecasts = []
    prix_dict = get_product_price_dict()

    for prod_id in df_history["ID_produit"].unique():
        df_prod = df_history[df_history["ID_produit"] == prod_id].copy()
        df_prod = creer_features_enrichies(df_prod)
        df_prod.sort_values("Date_commande", inplace=True)

        features = ['Jours', 'sin_jour_annee', 'cos_jour_annee',
                    'sin_mois', 'cos_mois', 'jour_semaine']

        X = df_prod[features].values
        y = df_prod["Quantite"].values.reshape(-1, 1)

        # Normalisation
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Logique d'entraînement et de prévision... (Identique à votre code)
        model = Sequential([
            LSTM(64, activation="relu", return_sequences=True, input_shape=(1, X.shape[1])),
            Dropout(0.2),
            LSTM(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=['mae'])
        early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

        if len(X_lstm) > 30:
            split_idx = int(len(X_lstm) * 0.8)
            X_train = X_lstm[:split_idx]
            y_train = y_scaled[:split_idx]
            model.fit(
                X_train, y_train, epochs=150, batch_size=16, verbose=0, callbacks=[early_stop]
            )
        else:
            model.fit(
                X_lstm, y_scaled, epochs=150, batch_size=16, verbose=0, callbacks=[early_stop]
            )

        # Prévisions futures
        last_date = df_prod["Date_commande"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        df_future = pd.DataFrame({'Date_commande': future_dates})
        df_future = creer_features_enrichies(df_future)
        last_day = df_prod['Jours'].max()
        df_future['Jours'] = [last_day + i for i in range(1, horizon + 1)]

        X_future = df_future[features].values
        X_future_scaled = scaler_X.transform(X_future)
        X_future_lstm = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))

        y_pred_scaled = model.predict(X_future_lstm, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_pred = np.maximum(y_pred, 0)

        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "ID_produit": prod_id,
            "Prévision": y_pred
        }))

    df_forecast = pd.concat(forecasts, ignore_index=True)
    df_forecast = calculate_chiffre_affaire(df_forecast, prix_dict)
    return df_forecast


# -----------------------------
# APPLICATION STREAMLIT
# -----------------------------

def main():
    st.set_page_config(layout="wide")
    st.title("Système de Prévision ML/DL - IA Logistique Durable")
    st.markdown("---")

    # --- Sidebar pour les entrées utilisateur ---
    st.sidebar.header("⚙️ Paramètres de Prévision")

    # Entrée de la durée de prévision
    horizon = st.sidebar.number_input(
        "⏱️ Durée de prévision (en jours) :",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )

    # Choix du modèle
    model_choice = st.sidebar.radio(
        "🧠 Choix du Modèle :",
        ["XGBoost Optimisé (ML)", "LSTM Optimisé (Deep Learning)"]
    )

    if st.sidebar.button("Lancer la Prévision"):

        st.info(f"⏳ Chargement des données historiques et entraînement des modèles pour {horizon} jours...")

        try:
            # Chargement des données
            df_history = load_data("historique_commandes")
            st.sidebar.success(f"📊 {len(df_history)} commandes historiques chargées.")

            # --- Exécution du Modèle Sélectionné ---
            if model_choice == "XGBoost Optimisé (ML)":
                model_name = "XGBoost"
                with st.spinner(f"Entraînement et prévision {model_name}..."):
                    df_forecast_res = forecast_xgb(df_history, horizon=horizon)
            else:
                model_name = "LSTM"
                with st.spinner(f"Entraînement et prévision {model_name}..."):
                    df_forecast_res = forecast_lstm(df_history, horizon=horizon)

            st.success(f"✅ Prévision {model_name} terminée pour {horizon} jours.")

            # --- Affichage des Résultats ---
            st.header(f"📈 Résultats de la Prévision {model_name}")

            # Résumé Chiffre d'Affaires
            summary = get_forecast_summary(df_forecast_res)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Unités Prévision", summary["Total_unites"])
            col2.metric("Moyenne Unités / Période", summary["Moyenne_unites"])
            col3.metric("Total Chiffre d'Affaires", summary["Total_CA"])
            col4.metric("Moyenne CA / Période", summary["Moyenne_CA"])

            st.markdown("---")

            # Tableau des Prévisions détaillées
            st.subheader("Détail des Prévisions (Unités & Chiffre d'Affaires)")
            st.dataframe(df_forecast_res.style.format({
                'Prévision': '{:,.2f}',
                'Chiffre_affaire': '{:,.2f} €'
            }), use_container_width=True)

            # Bouton de téléchargement CSV
            csv_file = df_forecast_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Télécharger Prévisions {model_name} (.csv)",
                data=csv_file,
                file_name=f'forecast_{model_name}_{horizon}jours.csv',
                mime='text/csv',
            )

            st.markdown("---")

            # Graphiques de Prévision
            st.subheader("Graphiques de Prévision par Produit")
            figures = plot_forecast_st(df_forecast_res, model_name, f"{horizon}j")

            # Afficher les figures dans des colonnes pour une meilleure mise en page
            cols_plot = st.columns(2)
            for i, fig in enumerate(figures):
                cols_plot[i % 2].pyplot(fig)  # Afficher dans une colonne sur deux

        except sqlite3.OperationalError:
            st.error(
                f"❌ Erreur de connexion à la base de données : Le fichier '{DB_PATH}' est-il au bon endroit et contient-il les tables 'historique_commandes' et 'produits' ?")
        except Exception as e:
            st.error(f"❌ Une erreur est survenue pendant l'exécution : {e}")


if __name__ == "__main__":
    # st.set_page_config(page_title="Prévision ML/DL", layout="wide") # Si non défini dans main
    main()
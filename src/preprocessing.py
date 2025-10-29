# src/preprocessing.py
"""
Module de prétraitement des données
Projet : IA_Logistique_Durable
Usage : from src.preprocessing import DataPreprocessor
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from src.database import Database  # ta classe pour accéder à la DB

class DataPreprocessor:
    """Classe pour le nettoyage et la transformation des données logistiques"""

    def __init__(self):
        pass

    # -----------------------------
    # CHARGEMENT DES DONNÉES DEPUIS LA BASE
    # -----------------------------
    @staticmethod
    def load_data_from_db() -> Dict[str, pd.DataFrame]:
        """Charge toutes les tables de la base dans des DataFrames"""
        data = {}
        db_path = "data/prevision.db"  # Chemin vers ta DB
        db = Database(db_path)
        db.connect()

        # Récupération des tables principales
        tables = {
            'fournisseurs': "SELECT * FROM fournisseurs",
            'transport': "SELECT * FROM transport",
            'inventaire': "SELECT * FROM produits",
            'commandes': "SELECT * FROM commandes",
            'expeditions': "SELECT * FROM expeditions"
        }

        for name, query in tables.items():
            result = db.query(query)
            data[name] = pd.DataFrame(result)

        db.disconnect()
        return data

    # -----------------------------
    # NETTOYAGE DES DONNÉES
    # -----------------------------
    @staticmethod
    def clean_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        df_clean = df.copy()
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        return df_clean

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        df_clean = df.copy()
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'fill_zero':
            df_clean = df_clean.fillna(0)
        elif strategy == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        return df_clean

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        return df.drop_duplicates(subset=subset, keep='first')

    # -----------------------------
    # TRANSFORMATION DES DONNÉES
    # -----------------------------
    @staticmethod
    def normalize_column(df: pd.DataFrame, column: str, method: str = 'minmax') -> pd.DataFrame:
        df_norm = df.copy()
        if column not in df_norm.columns:
            return df_norm
        if method == 'minmax':
            min_val = df_norm[column].min()
            max_val = df_norm[column].max()
            df_norm[f"{column}_normalized"] = (df_norm[column] - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0
        elif method == 'zscore':
            mean = df_norm[column].mean()
            std = df_norm[column].std()
            df_norm[f"{column}_normalized"] = (df_norm[column] - mean) / std if std != 0 else 0
        return df_norm

    @staticmethod
    def encode_categorical(df: pd.DataFrame, column: str, method: str = 'onehot') -> pd.DataFrame:
        df_encoded = df.copy()
        if column not in df_encoded.columns:
            return df_encoded
        if method == 'onehot':
            dummies = pd.get_dummies(df_encoded[column], prefix=column)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        elif method == 'label':
            categories = df_encoded[column].unique()
            df_encoded[f"{column}_encoded"] = df_encoded[column].map({cat: i for i, cat in enumerate(categories)})
        elif method == 'frequency':
            freq_map = df_encoded[column].value_counts(normalize=True).to_dict()
            df_encoded[f"{column}_freq"] = df_encoded[column].map(freq_map)
        return df_encoded

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Features logistiques et durables : délai, stock/commande, CO2, coût total"""
        df = df.copy()
        if 'Date_commande' in df.columns and 'Date_livraison_souhaitee' in df.columns:
            df['Delai_livraison_jours'] = (pd.to_datetime(df['Date_livraison_souhaitee']) - pd.to_datetime(df['Date_commande'])).dt.days
        if 'Quantite_disponible' in df.columns and 'Quantite_commandee' in df.columns:
            df['Stock_vs_commande'] = df['Quantite_disponible'] / df['Quantite_commandee'].replace(0, 1)
        if 'Quantite_commandee' in df.columns and 'Emissions_CO2_par_unite' in df.columns:
            df['CO2_total'] = df['Quantite_commandee'] * df['Emissions_CO2_par_unite']
        if 'Quantite_commandee' in df.columns and 'Cout_par_unite' in df.columns:
            df['Cout_total_estime'] = df['Quantite_commandee'] * df['Cout_par_unite']
        return df

    # -----------------------------
    # PRÉPARATION POUR MACHINE LEARNING
    # -----------------------------
    @staticmethod
    def prepare_for_ml(df: pd.DataFrame, target_column: str,
                       feature_columns: Optional[List[str]] = None,
                       test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df_clean = df.dropna(subset=[target_column])
        if feature_columns is None:
            feature_columns = [col for col in df_clean.columns if col != target_column]
        X = df_clean[feature_columns].select_dtypes(include=[np.number])
        y = df_clean[target_column]
        X = X.fillna(X.mean())
        split_index = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        return X_train, X_test, y_train, y_test

    # -----------------------------
    # PIPELINE COMPLET
    # -----------------------------
    @classmethod
    def preprocess_pipeline(cls, table_name: str,
                            clean_duplicates: bool = True,
                            handle_missing: str = 'drop',
                            normalize_numeric: bool = False) -> pd.DataFrame:
        data = cls.load_data_from_db()
        if table_name not in data:
            raise ValueError(f"Table '{table_name}' non trouvée")
        df = data[table_name]

        # Dates
        date_columns_map = {
            'inventaire': [],
            'commandes': ['Date_commande', 'Date_livraison_souhaitee'],
            'expeditions': ['Date_expedition', 'Date_livraison_estimee']
        }
        if table_name in date_columns_map:
            df = cls.clean_dates(df, date_columns_map[table_name])

        if clean_duplicates:
            df = cls.remove_duplicates(df)
        df = cls.handle_missing_values(df, strategy=handle_missing)

        # Features logistiques et durables
        df = cls.create_features(df)

        if normalize_numeric:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df = cls.normalize_column(df, col, method='minmax')

        return df


# -----------------------------
# RAPPORT DE QUALITÉ DES DONNÉES
# -----------------------------
def get_data_quality_report(df: pd.DataFrame) -> Dict:
    report = {
        'nombre_lignes': len(df),
        'nombre_colonnes': len(df.columns),
        'valeurs_manquantes': df.isnull().sum().to_dict(),
        'taux_completion': ((1 - df.isnull().sum() / len(df)) * 100).to_dict() if len(df) > 0 else {},
        'doublons': df.duplicated().sum(),
        'types_colonnes': df.dtypes.astype(str).to_dict()
    }
    return report

def print_data_quality_report(df: pd.DataFrame, table_name: str = "DataFrame"):
    report = get_data_quality_report(df)
    print(f"\n{'='*80}\n📊 RAPPORT DE QUALITÉ - {table_name}\n{'='*80}")
    print(f"📏 Dimensions: {report['nombre_lignes']} lignes × {report['nombre_colonnes']} colonnes")
    print(f"🔄 Doublons: {report['doublons']}")
    print(f"\n❓ Valeurs manquantes:")
    for col, count in report['valeurs_manquantes'].items():
        if count > 0:
            completion = report['taux_completion'].get(col, 0)
            print(f"   • {col}: {count} ({completion:.1f}% complet)")
    print()

"""
Module de prétraitement et rapport de qualité des données
Projet : IA_Logistique_Durable
Usage : python -m src.preprocessing
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional

DB_PATH = "data/prevision.db"

TABLE_QUERIES = {
    'fournisseurs': "SELECT * FROM Fournisseurs",
    'transport': "SELECT * FROM Transport",
    'inventaire': "SELECT * FROM Inventaire",
    'commandes': "SELECT * FROM Commandes",
    'expeditions': "SELECT * FROM Expeditions"
}

DATE_COLUMNS = {
    'commandes': ['Date_commande', 'Date_livraison_souhaitee'],
    'expeditions': ['Date_expedition', 'Date_livraison_estimee'],
    'inventaire': ['Date_derniere_mise_a_jour']
}


class DataPreprocessor:
    """Classe pour le nettoyage, transformation et rapport de données"""

    @staticmethod
    def load_data_from_db() -> Dict[str, pd.DataFrame]:
        data = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            for name, query in TABLE_QUERIES.items():
                data[name] = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as e:
            print(f"❌ Erreur lors du chargement de la DB : {e}")
        return data

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

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Date_commande' in df.columns and 'Date_livraison_souhaitee' in df.columns:
            df['Delai_livraison_jours'] = (pd.to_datetime(df['Date_livraison_souhaitee']) -
                                           pd.to_datetime(df['Date_commande'])).dt.days
        if 'Quantite_disponible' in df.columns and 'Quantite_commandee' in df.columns:
            df['Stock_vs_commande'] = df['Quantite_disponible'] / df['Quantite_commandee'].replace(0, 1)
        if 'Quantite_commandee' in df.columns and 'Emissions_CO2_par_livraison' in df.columns:
            df['CO2_total'] = df['Quantite_commandee'] * df['Emissions_CO2_par_livraison']
        if 'Quantite_commandee' in df.columns and 'Cout_par_unite' in df.columns:
            df['Cout_total_estime'] = df['Quantite_commandee'] * df['Cout_par_unite']
        return df

    @classmethod
    def preprocess_table(cls, df: pd.DataFrame, table_name: str,
                         handle_missing: str = 'drop',
                         clean_duplicates: bool = True) -> pd.DataFrame:
        if table_name in DATE_COLUMNS:
            df = cls.clean_dates(df, DATE_COLUMNS[table_name])
        if clean_duplicates:
            df = cls.remove_duplicates(df)
        df = cls.handle_missing_values(df, handle_missing)
        df = cls.create_features(df)
        return df


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


def print_data_quality_report(df: pd.DataFrame, table_name: str):
    report = get_data_quality_report(df)
    print(f"\n{'='*80}\n📊 RAPPORT DE QUALITÉ - {table_name}\n{'='*80}")
    print(f"📏 Dimensions : {report['nombre_lignes']} lignes × {report['nombre_colonnes']} colonnes")
    print(f"🔄 Doublons : {report['doublons']}")
    print(f"\n❓ Valeurs manquantes :")
    for col, count in report['valeurs_manquantes'].items():
        if count > 0:
            completion = report['taux_completion'].get(col, 0)
            print(f"   • {col} : {count} ({completion:.1f}% complet)")
    print(f"\n🧪 Types de colonnes : {report['types_colonnes']}\n")


# -----------------------------
# SCRIPT PRINCIPAL
# -----------------------------
if __name__ == "__main__":
    print("\n🚀 Lancement du prétraitement et rapport de qualité\n")
    data = DataPreprocessor.load_data_from_db()
    for table_name, df in data.items():
        print(f"\n--- Traitement de la table '{table_name}' ---")
        df_clean = DataPreprocessor.preprocess_table(df, table_name)
        print_data_quality_report(df_clean, table_name)

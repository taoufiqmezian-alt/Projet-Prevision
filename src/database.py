"""
Module Database pour le projet IA_Logistique_Durable
----------------------------------------------------
Permet :
- la connexion à la base SQLite
- l’exécution de requêtes SQL
- l’affichage de tables et statistiques
"""

import sqlite3
import pandas as pd
from tabulate import tabulate


class Database:
    def __init__(self, db_path):
        """Initialise la classe avec le chemin de la base de données."""
        self.db_path = db_path
        self.conn = None

    # --------------------------------------------------
    # Connexion / Déconnexion
    # --------------------------------------------------
    def connect(self):
        """Établit la connexion à la base de données"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            print(f"✅ Connexion réussie à {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Erreur de connexion : {e}")

    def disconnect(self):
        """Ferme la connexion à la base de données"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("🔌 Connexion fermée")

    # --------------------------------------------------
    # Requêtes SQL
    # --------------------------------------------------
    def execute(self, sql, params=()):
        """Exécute une requête SQL (INSERT, UPDATE, DELETE)"""
        try:
            if self.conn is None:
                self.connect()
            cur = self.conn.cursor()
            cur.execute(sql, params)
            self.conn.commit()
            return cur.rowcount
        except sqlite3.Error as e:
            print(f"❌ Erreur d'exécution : {e}")
            return 0

    def query(self, sql, params=()):
        """Exécute une requête SQL SELECT et retourne le résultat"""
        try:
            if self.conn is None:
                self.connect()
            cur = self.conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            if rows and isinstance(rows[0], sqlite3.Row):
                return [dict(row) for row in rows]
            return rows
        except sqlite3.Error as e:
            print(f"❌ Erreur de requête : {e}")
            return []

    def get_table_columns(self, table_name):
        """Récupère les noms des colonnes d'une table"""
        rows = self.query(f"PRAGMA table_info({table_name})")
        return [row['name'] for row in rows] if rows else []


# ==================================================
# FONCTIONS UTILITAIRES : affichage de données
# ==================================================
def afficher_tableau(db_path, table_name, limit=10):
    """Affiche les premières lignes d'une table sous forme de DataFrame."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        conn.close()
        print(f"\n=== Aperçu de la table '{table_name}' ===")
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
        return df
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage de la table : {e}")


def afficher_statistiques(db_path, table_name):
    """Affiche des statistiques de base sur une table."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        print(f"\n📊 Statistiques de la table '{table_name}':")
        print(df.describe(include='all'))
        return df
    except Exception as e:
        print(f"❌ Erreur lors du calcul des statistiques : {e}")


# ==================================================
# TEST LOCAL (facultatif)
# ==================================================
if __name__ == "__main__":
    db_path = "data/prevision.db"
    print("\n🧪 Test de la classe Database...\n")
    db = Database(db_path)
    db.connect()

    # Exemple : afficher quelques produits
    afficher_tableau(db_path, "produits", limit=5)

    # Exemple : statistiques sur les commandes
    afficher_statistiques(db_path, "historique_commandes")

    db.disconnect()

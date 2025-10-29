"""
Module Database pour le projet IA_Logistique_Durable
Usage : import Database et fonctions associées
"""

import sqlite3
from tabulate import tabulate


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

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

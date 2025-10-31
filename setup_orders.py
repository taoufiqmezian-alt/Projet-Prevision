import sqlite3
import pandas as pd

DB_PATH = "data/prevision.db"

# Connexion à la base de données
conn = sqlite3.connect(DB_PATH)
print("✅ Connexion réussie à la base de données")

# Création de la table historique_commandes
conn.execute("""
CREATE TABLE IF NOT EXISTS historique_commandes (
    ID_commande INTEGER PRIMARY KEY AUTOINCREMENT,
    Date_commande TEXT NOT NULL,
    ID_produit TEXT NOT NULL,
    Quantite INTEGER NOT NULL,
    FOREIGN KEY (ID_produit) REFERENCES produits (ID_produit)
)
""")

# Quelques données de test (3 mois d'historique)
data = [
    ("2025-07-01", "P001", 120),
    ("2025-07-15", "P002", 200),
    ("2025-07-30", "P003", 150),
    ("2025-08-05", "P001", 130),
    ("2025-08-20", "P002", 210),
    ("2025-08-30", "P004", 180),
    ("2025-09-05", "P003", 160),
    ("2025-09-18", "P005", 90),
    ("2025-09-25", "P006", 75),
    ("2025-09-30", "P007", 300)
]

# Conversion en DataFrame pour insertion facile
df = pd.DataFrame(data, columns=["Date_commande", "ID_produit", "Quantite"])
df.to_sql("historique_commandes", conn, if_exists="replace", index=False)

print("✅ Table 'historique_commandes' créée avec succès et peuplée de données !")

conn.close()
print("🔌 Connexion fermée")

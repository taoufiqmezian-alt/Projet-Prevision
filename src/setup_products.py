# src/setup_products.py
from src.database import Database

db_path = "data/prevision.db"
db = Database(db_path)
db.connect()

# Créer la table produits si elle n'existe pas
db.execute("""
CREATE TABLE IF NOT EXISTS produits (
    ID_produit TEXT PRIMARY KEY,
    Nom_produit TEXT,
    Quantite_disponible INTEGER,
    Cout_par_unite REAL,
    Emissions_CO2_par_unite REAL
)
""")

# Liste complète des produits
produits = [
    ('P001', 'SteelBeam GreenSteel', 100, 5.0, 10.0),
    ('P002', 'EcoPlastic Sheet', 200, 3.0, 5.0),
    ('P003', 'BioWood Panel', 150, 4.5, 8.0),
    ('P004', 'RecyGlass Bottle', 250, 2.5, 2.0),
    ('P005', 'Solar Panel Module', 50, 50.0, 20.0),
    ('P006', 'AluFrame Structure', 80, 12.0, 15.0),
    ('P007', 'Battery Pack Lithium', 120, 30.0, 25.0)
]

# Insérer chaque produit dans la table
for p in produits:
    db.execute("INSERT OR IGNORE INTO produits VALUES (?, ?, ?, ?, ?)", p)

db.disconnect()
print("✅ Table produits créée et 7 produits insérés")

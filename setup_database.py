"""
Script de création de la base de données prevision.db
Projet : IA_Logistique_Durable (Projet prévision)
Structure : data/prevision.db
Usage : python setup_database.py
"""

import sqlite3
import os
from datetime import datetime


def create_database():
    """
    Crée la base de données prevision.db avec toutes les tables
    et insère les données initiales
    """

    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)

    # Connexion (ou création) à la base SQLite
    db_path = "data/prevision.db"  # Sans accent pour compatibilité

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Activer les clés étrangères
            cursor.execute("PRAGMA foreign_keys = ON;")

            print("=" * 70)
            print("🚀 CRÉATION DE LA BASE DE DONNÉES - Projet prévision")
            print("=" * 70)
            print(f"📂 Emplacement : {db_path}")
            print(f"📅 Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            # =======================================================
            # TABLE 1 : Fournisseurs
            # =======================================================
            print("📊 Création de la table 'Fournisseurs'...")
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Fournisseurs
                           (
                               ID_fournisseur
                               TEXT
                               PRIMARY
                               KEY,
                               Nom_fournisseur
                               TEXT
                               NOT
                               NULL,
                               Adresse_fournisseur
                               TEXT,
                               Type_de_fourniture
                               TEXT,
                               Cout_par_unite
                               REAL
                               CHECK
                           (
                               Cout_par_unite >
                               0
                           ),
                               Emissions_CO2_par_livraison REAL CHECK
                           (
                               Emissions_CO2_par_livraison
                               >=
                               0
                           )
                               )
                           """)

            fournisseurs_data = [
                ('F001', 'GreenSteel', '12 rue de l\'Industrie, Lyon, France', 'Matières premières', 5.5, 120),
                ('F002', 'EcoPlast', '45 avenue Verte, Paris, France', 'Composants plastiques', 2.8, 80),
                ('F003', 'BioWood', '8 rue du Bois, Toulouse, France', 'Emballages bois', 3.2, 60),
                ('F004', 'RecyGlass', '10 rue du Verre, Marseille, France', 'Verre recyclé', 4.5, 100),
                ('F005', 'SolarParts', '22 rue des Électroniques, Grenoble, France', 'Composants électroniques', 7.2,
                 150),
                ('F006', 'EcoTextile', '15 rue du Coton, Lille, France', 'Textiles durables', 6.0, 90),
                ('F007', 'BioChem', '30 avenue des Sciences, Nantes, France', 'Produits chimiques bio', 8.5, 130),
                ('F008', 'RecyPaper', '7 rue du Papier, Bordeaux, France', 'Papier recyclé', 3.5, 70)
            ]
            cursor.executemany("INSERT OR IGNORE INTO Fournisseurs VALUES (?, ?, ?, ?, ?, ?)", fournisseurs_data)
            print(f"   ✅ {len(fournisseurs_data)} fournisseurs insérés")

            # =======================================================
            # TABLE 2 : Transport
            # =======================================================
            print("📊 Création de la table 'Transport'...")
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Transport
                           (
                               ID_transport
                               TEXT
                               PRIMARY
                               KEY,
                               Type_vehicule
                               TEXT
                               NOT
                               NULL,
                               Capacite_maximale
                               INTEGER
                               CHECK
                           (
                               Capacite_maximale >
                               0
                           ),
                               Consommation_carburant REAL CHECK
                           (
                               Consommation_carburant
                               >=
                               0
                           ),
                               Emissions_CO2_par_km REAL CHECK
                           (
                               Emissions_CO2_par_km
                               >=
                               0
                           ),
                               Cout_par_km REAL CHECK
                           (
                               Cout_par_km >
                               0
                           ),
                               Disponibilite TEXT
                               )
                           """)

            transport_data = [
                ('TR001', 'Camion diesel', 19000, 28, 0.90, 1.25, 'Tous les jours 06h–22h'),
                ('TR002', 'Camionnette diesel', 3500, 9, 0.28, 0.65, 'Tous les jours 08h–20h'),
                ('TR003', 'Camion électrique', 12000, 110, 0.08, 1.60, 'Lun–Ven 07h–18h'),
                ('TR004', 'Train fret', 60000, 0, 0.03, 0.45, 'Selon horaire SNCF Fret'),
                ('TR005', 'Bateau cargo moyen', 500000, 0, 0.20, 0.25, 'Sur demande (export maritime)'),
                ('TR006', 'Véhicule utilitaire GNV', 2000, 7, 0.20, 0.55, 'Tous les jours 07h–21h'),
                ('TR007', 'Camion hybride', 14000, 20, 0.60, 1.10, 'Lun–Sam 06h–22h'),
                ('TR008', 'Vélo cargo électrique', 200, 4, 0.01, 0.15, 'Lun–Ven 08h–18h')
            ]
            cursor.executemany("INSERT OR IGNORE INTO Transport VALUES (?, ?, ?, ?, ?, ?, ?)", transport_data)
            print(f"   ✅ {len(transport_data)} moyens de transport insérés")

            # =======================================================
            # TABLE 3 : Inventaire
            # =======================================================
            print("📊 Création de la table 'Inventaire'...")
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Inventaire
                           (
                               ID_produit
                               TEXT
                               PRIMARY
                               KEY,
                               Nom_produit
                               TEXT
                               NOT
                               NULL,
                               Quantite_disponible
                               INTEGER
                               CHECK
                           (
                               Quantite_disponible
                               >=
                               0
                           ),
                               Unite TEXT,
                               Emplacement_stock TEXT,
                               ID_fournisseur TEXT,
                               Date_derniere_mise_a_jour TEXT,
                               Seuil_minimum INTEGER CHECK
                           (
                               Seuil_minimum
                               >=
                               0
                           ),
                               Seuil_maximum INTEGER CHECK
                           (
                               Seuil_maximum >
                               Seuil_minimum
                           ),
                               Etat_produit TEXT,
                               FOREIGN KEY
                           (
                               ID_fournisseur
                           ) REFERENCES Fournisseurs
                           (
                               ID_fournisseur
                           )
                               )
                           """)

            inventaire_data = [
                ('P001', 'Acier recyclé', 12000, 'kg', 'Entrepôt Nord – Zone A1 – Rayonnage 3', 'F001', '2025-10-20',
                 5000, 20000, 'Bon état'),
                ('P002', 'Granulés plastique recyclé', 8500, 'kg', 'Entrepôt Sud – Zone B2 – Silos 1-2', 'F002',
                 '2025-10-22', 3000, 15000, 'Bon état'),
                ('P003', 'LED haute efficacité', 3200, 'pièce', 'Atelier Électronique – Rayonnage C4', 'F005',
                 '2025-10-24', 1000, 5000, 'Bon état'),
                ('P004', 'Cartons biodégradables', 15000, 'unité', 'Entrepôt Emballage – Zone E1', 'F008', '2025-10-21',
                 5000, 25000, 'Bon état'),
                ('P005', 'Lubrifiant écologique', 950, 'litre', 'Zone Maintenance – Rayonnage D2', 'F007', '2025-10-23',
                 300, 2000, 'Bon état'),
                ('P006', 'Verre industriel trempé', 4200, 'kg', 'Atelier Verrerie – Zone G3', 'F004', '2025-10-19',
                 2000, 8000, 'Bon état'),
                ('P007', 'Modules électroniques', 1250, 'pièce', 'Atelier Montage – Zone C2', 'F005', '2025-10-24', 500,
                 3000, 'Bon état'),
                ('P008', 'Rubans adhésifs recyclables', 3600, 'rouleau', 'Entrepôt Emballage – Zone E3', 'F008',
                 '2025-10-18', 1000, 6000, 'Bon état'),
                ('P009', 'Colle industrielle bio-basée', 750, 'litre', 'Zone Production – Rayonnage F1', 'F007',
                 '2025-10-20', 200, 1500, 'Bon état'),
                ('P010', 'Acier inoxydable (tiges)', 9800, 'kg', 'Entrepôt Nord – Zone A2', 'F001', '2025-10-22', 4000,
                 15000, 'Bon état')
            ]
            cursor.executemany("INSERT OR IGNORE INTO Inventaire VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               inventaire_data)
            print(f"   ✅ {len(inventaire_data)} produits insérés dans l'inventaire")

            # =======================================================
            # TABLE 4 : Commandes
            # =======================================================
            print("📊 Création de la table 'Commandes'...")
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Commandes
                           (
                               ID_commande
                               TEXT
                               PRIMARY
                               KEY,
                               ID_client
                               TEXT,
                               Date_commande
                               TEXT,
                               Date_livraison_souhaitee
                               TEXT,
                               ID_produit
                               TEXT,
                               Quantite_commandee
                               INTEGER
                               CHECK
                           (
                               Quantite_commandee >
                               0
                           ),
                               Statut_commande TEXT CHECK
                           (
                               Statut_commande
                               IN
                           (
                               'En attente',
                               'En cours',
                               'Livrée',
                               'Annulée'
                           )),
                               Cout_total REAL CHECK
                           (
                               Cout_total
                               >=
                               0
                           ),
                               FOREIGN KEY
                           (
                               ID_produit
                           ) REFERENCES Inventaire
                           (
                               ID_produit
                           )
                               )
                           """)

            commandes_data = [
                ('CMD001', 'CL001', '2025-10-10', '2025-10-18', 'P001', 3000, 'Livrée', 16500),
                ('CMD002', 'CL002', '2025-10-12', '2025-10-21', 'P004', 5000, 'En cours', 11250),
                ('CMD003', 'CL003', '2025-10-14', '2025-10-25', 'P003', 1200, 'En attente', 3360),
                ('CMD004', 'CL001', '2025-10-15', '2025-10-24', 'P002', 4000, 'En cours', 11200),
                ('CMD005', 'CL004', '2025-10-16', '2025-10-28', 'P007', 600, 'Livrée', 1680),
                ('CMD006', 'CL005', '2025-10-18', '2025-10-27', 'P005', 400, 'En attente', 5000),
                ('CMD007', 'CL002', '2025-10-20', '2025-10-30', 'P009', 300, 'En attente', 1950),
                ('CMD008', 'CL006', '2025-10-21', '2025-11-01', 'P010', 5000, 'En cours', 30000),
                ('CMD009', 'CL004', '2025-10-22', '2025-10-31', 'P006', 2500, 'En cours', 15500),
                ('CMD010', 'CL007', '2025-10-23', '2025-11-02', 'P008', 1800, 'En attente', 2430)
            ]
            cursor.executemany("INSERT OR IGNORE INTO Commandes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", commandes_data)
            print(f"   ✅ {len(commandes_data)} commandes insérées")

            # =======================================================
            # TABLE 5 : Expéditions
            # =======================================================
            print("📊 Création de la table 'Expeditions'...")
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS Expeditions
                           (
                               ID_expedition
                               TEXT
                               PRIMARY
                               KEY,
                               ID_commande
                               TEXT,
                               ID_transport
                               TEXT,
                               Date_expedition
                               TEXT,
                               Date_livraison_estimee
                               TEXT,
                               Statut_livraison
                               TEXT
                               CHECK (
                               Statut_livraison
                               IN
                           (
                               'En transit',
                               'Livrée',
                               'Retardée',
                               'Annulée'
                           )),
                               Distance_parcourue REAL CHECK
                           (
                               Distance_parcourue
                               >=
                               0
                           ),
                               Cout_transport REAL CHECK
                           (
                               Cout_transport
                               >=
                               0
                           ),
                               Emissions_CO2_effectives REAL CHECK
                           (
                               Emissions_CO2_effectives
                               >=
                               0
                           ),
                               FOREIGN KEY
                           (
                               ID_commande
                           ) REFERENCES Commandes
                           (
                               ID_commande
                           ),
                               FOREIGN KEY
                           (
                               ID_transport
                           ) REFERENCES Transport
                           (
                               ID_transport
                           )
                               )
                           """)

            expeditions_data = [
                ('EXP001', 'CMD001', 'TR001', '2025-10-01', '2025-10-03', 'Livrée', 320, 480, 250),
                ('EXP002', 'CMD002', 'TR002', '2025-10-02', '2025-10-05', 'Livrée', 150, 150, 75),
                ('EXP003', 'CMD003', 'TR003', '2025-10-03', '2025-10-07', 'Retardée', 820, 1200, 640),
                ('EXP004', 'CMD004', 'TR001', '2025-10-05', '2025-10-07', 'Livrée', 280, 420, 220),
                ('EXP005', 'CMD005', 'TR004', '2025-10-06', '2025-10-08', 'En transit', 560, 700, 310),
                ('EXP006', 'CMD006', 'TR002', '2025-10-07', '2025-10-09', 'Livrée', 190, 200, 95),
                ('EXP007', 'CMD007', 'TR005', '2025-10-10', '2025-10-12', 'Retardée', 1200, 2000, 980),
                ('EXP008', 'CMD008', 'TR001', '2025-10-11', '2025-10-14', 'En transit', 600, 900, 400)
            ]
            cursor.executemany("INSERT OR IGNORE INTO Expeditions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", expeditions_data)
            print(f"   ✅ {len(expeditions_data)} expéditions insérées")

            # =======================================================
            # CRÉATION DES INDEX POUR PERFORMANCES
            # =======================================================
            print("\n📊 Création des index pour optimiser les requêtes...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inventaire_fournisseur ON Inventaire(ID_fournisseur)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commandes_produit ON Commandes(ID_produit)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_commandes_statut ON Commandes(Statut_commande)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expeditions_commande ON Expeditions(ID_commande)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expeditions_transport ON Expeditions(ID_transport)")
            print("   ✅ Index créés avec succès")

            # Commit explicite
            conn.commit()

            # Vérification finale
            print()
            print("=" * 70)
            print("✅ VÉRIFICATION DES DONNÉES")
            print("=" * 70)
            for table in ["Fournisseurs", "Transport", "Inventaire", "Commandes", "Expeditions"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                print(f"📦 {table:<13}: {cursor.fetchone()[0]} enregistrements")

            print()
            print("=" * 70)
            print("🎉 BASE DE DONNÉES CRÉÉE AVEC SUCCÈS !")
            print("=" * 70)
            print(f"📂 Fichier : {db_path}")
            print(f"💾 Taille : {os.path.getsize(db_path) / 1024:.2f} Ko")
            print()
            print("🚀 Prochaine étape : Créer src/database.py pour interagir avec la DB")
            print()

    except sqlite3.Error as e:
        print(f"\n❌ ERREUR SQLite : {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    create_database()
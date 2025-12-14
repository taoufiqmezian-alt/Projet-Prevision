"""
setup_orders.py - Version enrichie avec événements
Inclut : Noël, Promotions, Jours fériés, Black Friday, Vacances
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

DB_PATH = "data/prevision.db"


# ============================================================
# CALENDRIER DES ÉVÉNEMENTS
# ============================================================

def get_jours_feries(annee):
    """Retourne les jours fériés français pour une année"""
    jours_feries = [
        f"{annee}-01-01",  # Nouvel An
        f"{annee}-05-01",  # Fête du Travail
        f"{annee}-05-08",  # Victoire 1945
        f"{annee}-07-14",  # Fête Nationale
        f"{annee}-08-15",  # Assomption
        f"{annee}-11-01",  # Toussaint
        f"{annee}-11-11",  # Armistice
        f"{annee}-12-25",  # Noël
    ]
    return [pd.to_datetime(d) for d in jours_feries]


def get_periodes_promotions(annee):
    """Retourne les périodes de promotions"""
    return [
        # Soldes d'hiver (2 semaines en janvier)
        (pd.to_datetime(f"{annee}-01-10"), pd.to_datetime(f"{annee}-01-24"), 1.4, "Soldes hiver"),

        # Black Friday (fin novembre)
        (pd.to_datetime(f"{annee}-11-24"), pd.to_datetime(f"{annee}-11-27"), 1.8, "Black Friday"),

        # Cyber Monday
        (pd.to_datetime(f"{annee}-11-28"), pd.to_datetime(f"{annee}-11-30"), 1.6, "Cyber Monday"),

        # Soldes d'été (2 semaines en juillet)
        (pd.to_datetime(f"{annee}-07-01"), pd.to_datetime(f"{annee}-07-15"), 1.3, "Soldes été"),

        # Rentrée scolaire (septembre)
        (pd.to_datetime(f"{annee}-09-01"), pd.to_datetime(f"{annee}-09-10"), 1.5, "Rentrée"),
    ]


def get_vacances_scolaires(annee):
    """Retourne les vacances scolaires (zones confondues)"""
    return [
        # Vacances d'hiver (février)
        (pd.to_datetime(f"{annee}-02-10"), pd.to_datetime(f"{annee}-02-25"), 0.7),

        # Vacances de printemps (avril)
        (pd.to_datetime(f"{annee}-04-15"), pd.to_datetime(f"{annee}-04-30"), 0.7),

        # Grandes vacances (juillet-août)
        (pd.to_datetime(f"{annee}-07-10"), pd.to_datetime(f"{annee}-08-25"), 0.6),

        # Vacances de Toussaint (octobre)
        (pd.to_datetime(f"{annee}-10-20"), pd.to_datetime(f"{annee}-11-03"), 0.8),
    ]


def est_jour_ferie(date, jours_feries):
    """Vérifie si la date est un jour férié"""
    return date in jours_feries


def est_en_promotion(date, promotions):
    """Vérifie si la date est en période de promotion"""
    for debut, fin, facteur, nom in promotions:
        if debut <= date <= fin:
            return facteur, nom
    return 1.0, None


def est_en_vacances(date, vacances):
    """Vérifie si la date est en vacances scolaires"""
    for debut, fin, facteur in vacances:
        if debut <= date <= fin:
            return facteur
    return 1.0


# ============================================================
# GÉNÉRATION DE DONNÉES
# ============================================================

def creer_table():
    """Crée la table historique_commandes"""
    conn = sqlite3.connect(DB_PATH)
    print("✅ Connexion réussie à la base de données")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS historique_commandes (
        ID_commande INTEGER PRIMARY KEY AUTOINCREMENT,
        Date_commande TEXT NOT NULL,
        ID_produit TEXT NOT NULL,
        Quantite INTEGER NOT NULL,
        FOREIGN KEY (ID_produit) REFERENCES produits (ID_produit)
    )
    """)
    print("✅ Table 'historique_commandes' créée")
    conn.close()


def generer_donnees_realistes(mois=18):
    """
    Génère des données avec événements réalistes

    Événements inclus :
    - ✅ Saisonnalité annuelle
    - ✅ Tendance croissante
    - ✅ Jours fériés (baisse activité)
    - ✅ Promotions (Black Friday, Soldes, Rentrée)
    - ✅ Vacances scolaires (baisse activité)
    - ✅ Pics de Noël
    - ✅ Variations jour de semaine
    """
    print(f"\n{'=' * 80}")
    print(f"🎉 GÉNÉRATION DE {mois} MOIS AVEC ÉVÉNEMENTS RÉALISTES")
    print(f"{'=' * 80}\n")

    # Configuration des produits
    produits_config = {
        'P001': {
            'nom': 'Acier recyclé',
            'base': 125,
            'saisonalite': 30,
            'tendance': 0.15,
            'sensible_promo': False,      # Peu sensible aux promos
            'sensible_vacances': False
        },
        'P002': {
            'nom': 'Granulés plastique',
            'base': 205,
            'saisonalite': 35,
            'tendance': 0.10,
            'sensible_promo': False,
            'sensible_vacances': False
        },
        'P003': {
            'nom': 'LED',
            'base': 155,
            'saisonalite': 25,
            'tendance': 0.05,
            'sensible_promo': True,        # Sensible aux promos
            'sensible_vacances': True
        },
        'P004': {
            'nom': 'Cartons',
            'base': 180,
            'saisonalite': 50,
            'tendance': 0.08,
            'sensible_promo': True,        # Très sensible (e-commerce)
            'sensible_vacances': True
        },
        'P005': {
            'nom': 'Lubrifiant',
            'base': 90,
            'saisonalite': 15,
            'tendance': 0.02,
            'sensible_promo': False,
            'sensible_vacances': False
        },
        'P006': {
            'nom': 'Verre',
            'base': 75,
            'saisonalite': 20,
            'tendance': 0.12,
            'sensible_promo': True,
            'sensible_vacances': False
        },
        'P007': {
            'nom': 'Modules électroniques',
            'base': 300,
            'saisonalite': 60,
            'tendance': 0.20,
            'sensible_promo': True,        # Très sensible
            'sensible_vacances': True
        }
    }

    # Dates
    date_fin = datetime.now()
    date_debut = date_fin - timedelta(days=mois * 30)

    # Générer dates (3 commandes par semaine)
    dates = pd.date_range(start=date_debut, end=date_fin, freq='2.33D')

    # Calendrier des événements
    annees = [2023, 2024, 2025]
    jours_feries = []
    promotions = []
    vacances = []

    for annee in annees:
        jours_feries.extend(get_jours_feries(annee))
        promotions.extend(get_periodes_promotions(annee))
        vacances.extend(get_vacances_scolaires(annee))

    donnees = []
    evenements_compteur = {
        'jours_feries': 0,
        'promotions': 0,
        'vacances': 0,
        'noel': 0
    }

    for prod_id, config in produits_config.items():
        print(f"   Génération {prod_id} - {config['nom']}...", end='')

        for i, date in enumerate(dates):
            # === COMPOSANTES DE BASE ===
            base = config['base']

            # 1. Tendance
            tendance = base * config['tendance'] * (i / len(dates))

            # 2. Saisonnalité annuelle
            jour_annee = date.timetuple().tm_yday
            saison = config['saisonalite'] * np.sin(2 * np.pi * jour_annee / 365.25 - np.pi/2)

            # 3. Jour de la semaine
            jour_semaine = date.weekday()
            if jour_semaine <= 2:  # Lun-Mer
                facteur_jour = random.uniform(1.05, 1.20)
            elif jour_semaine <= 4:  # Jeu-Ven
                facteur_jour = random.uniform(0.95, 1.10)
            else:  # Weekend
                facteur_jour = random.uniform(0.50, 0.75)

            # === ÉVÉNEMENTS SPÉCIAUX ===

            # 4. Jours fériés (baisse activité)
            if est_jour_ferie(date, jours_feries):
                facteur_ferie = 0.3  # -70%
                evenements_compteur['jours_feries'] += 1
            else:
                facteur_ferie = 1.0

            # 5. Promotions (augmentation demande)
            facteur_promo, nom_promo = est_en_promotion(date, promotions)
            if nom_promo and config['sensible_promo']:
                evenements_compteur['promotions'] += 1
            elif not config['sensible_promo']:
                facteur_promo = 1.0  # Produit non sensible aux promos

            # 6. Vacances scolaires (baisse pour certains produits)
            facteur_vacances = est_en_vacances(date, vacances)
            if facteur_vacances < 1.0 and not config['sensible_vacances']:
                facteur_vacances = 1.0  # Produit non affecté par vacances

            # 7. Pic de fin d'année (Noël)
            if date.month == 12:
                if prod_id in ['P004', 'P007']:  # Cartons + Électronique
                    facteur_noel = random.uniform(1.4, 1.7)
                    evenements_compteur['noel'] += 1
                else:
                    facteur_noel = random.uniform(1.1, 1.2)
            elif date.month == 11 and prod_id == 'P004':
                facteur_noel = random.uniform(1.2, 1.4)
            else:
                facteur_noel = 1.0

            # 8. Bruit aléatoire
            bruit = np.random.normal(0, base * 0.12)

            # 9. Événements rares (pics/creux aléatoires)
            if random.random() < 0.03:
                evenement = random.choice([1.6, 0.4])
            else:
                evenement = 1.0

            # === CALCUL FINAL ===
            quantite = base + tendance + saison + bruit
            quantite = quantite * facteur_jour * facteur_ferie * facteur_promo
            quantite = quantite * facteur_vacances * facteur_noel * evenement
            quantite = max(30, int(quantite))

            donnees.append({
                'Date_commande': date.strftime('%Y-%m-%d'),
                'ID_produit': prod_id,
                'Quantite': quantite
            })

        print(f" ✓ ({len(dates)} points)")

    df = pd.DataFrame(donnees)

    # Statistiques
    print(f"\n{'=' * 80}")
    print(f"✅ Génération terminée")
    print(f"📊 Total : {len(df):,} commandes")
    print(f"📅 Période : {df['Date_commande'].min()} → {df['Date_commande'].max()}")
    print(f"📈 Par produit : {len(df) // len(produits_config):,} en moyenne")

    print(f"\n🎉 Événements inclus :")
    print(f"   • Jours fériés : {evenements_compteur['jours_feries']} occurrences")
    print(f"   • Promotions : {evenements_compteur['promotions']} occurrences")
    print(f"   • Pics Noël : {evenements_compteur['noel']} occurrences")
    print(f"   • Vacances : Impact sur produits sensibles")
    print(f"{'=' * 80}\n")

    return df


def inserer_donnees(df, mode='replace'):
    """Insère les données dans la base"""
    conn = sqlite3.connect(DB_PATH)

    # Backup si remplacement
    if mode == 'replace':
        try:
            df_old = pd.read_sql_query("SELECT * FROM historique_commandes", conn)
            if len(df_old) > 0:
                backup_file = f"backup_historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_old.to_csv(backup_file, index=False)
                print(f"💾 Backup créé : {backup_file}")
        except:
            pass

    # Insertion
    df.to_sql('historique_commandes', conn, if_exists=mode, index=False)

    # Vérification
    result = pd.read_sql_query("SELECT COUNT(*) as total FROM historique_commandes", conn)
    total = result['total'].iloc[0]

    print(f"✅ Données insérées avec succès !")
    print(f"📊 Total dans la base : {total:,} enregistrements\n")

    conn.close()


def afficher_apercu():
    """Affiche un aperçu des données"""
    conn = sqlite3.connect(DB_PATH)

    print(f"{'=' * 80}")
    print("📋 APERÇU DES DONNÉES")
    print(f"{'=' * 80}\n")

    # Premières lignes
    df = pd.read_sql_query("SELECT * FROM historique_commandes LIMIT 5", conn)
    print("Premières lignes :")
    print(df.to_string(index=False))

    # Statistiques par produit
    print(f"\n📊 Statistiques par produit :")
    stats = pd.read_sql_query("""
        SELECT 
            ID_produit,
            COUNT(*) as nb_commandes,
            AVG(Quantite) as moy,
            MIN(Quantite) as min,
            MAX(Quantite) as max,
            SUM(Quantite) as total
        FROM historique_commandes
        GROUP BY ID_produit
        ORDER BY ID_produit
    """, conn)

    for _, row in stats.iterrows():
        print(f"\n   {row['ID_produit']}:")
        print(f"      • Commandes : {int(row['nb_commandes'])}")
        print(f"      • Moyenne   : {row['moy']:.1f}")
        print(f"      • Min → Max : {int(row['min'])} → {int(row['max'])}")
        print(f"      • Total     : {int(row['total']):,}")

    print(f"\n{'=' * 80}\n")

    conn.close()


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("   🚀 SETUP HISTORIQUE COMMANDES - VERSION ÉVÉNEMENTS")
    print("=" * 80)
    print("   ✅ Jours fériés | ✅ Promotions | ✅ Noël | ✅ Vacances")
    print("=" * 80)

    # Étape 1 : Créer la table
    creer_table()

    # Étape 2 : Choix utilisateur
    print("\n📋 OPTIONS DE GÉNÉRATION")
    print("-" * 80)
    print("1️⃣  Données de test (10 commandes)")
    print("2️⃣  Données réalistes - 12 mois + événements")
    print("3️⃣  Données réalistes - 18 mois + événements (RECOMMANDÉ)")
    print("4️⃣  Données réalistes - 24 mois + événements")

    choix = input("\n➡️  Choisissez une option (1-4) : ").strip()

    if choix == '1':
        # Données de test minimales
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
        df = pd.DataFrame(data, columns=["Date_commande", "ID_produit", "Quantite"])
        print("\n⚠️  Mode test : 10 commandes uniquement (sans événements)")

    elif choix == '2':
        df = generer_donnees_realistes(mois=12)
    elif choix == '3':
        df = generer_donnees_realistes(mois=18)
    elif choix == '4':
        df = generer_donnees_realistes(mois=24)
    else:
        print("⚠️  Choix invalide, génération de 18 mois par défaut")
        df = generer_donnees_realistes(mois=18)

    # Étape 3 : Insertion
    inserer_donnees(df, mode='replace')

    # Étape 4 : Aperçu
    afficher_apercu()

    # Fin
    print("=" * 80)
    print("   ✅ CONFIGURATION TERMINÉE AVEC ÉVÉNEMENTS")
    print("=" * 80)
    print("\n🎉 Événements inclus :")
    print("   ✅ Jours fériés (8 par an)")
    print("   ✅ Black Friday / Cyber Monday")
    print("   ✅ Soldes (hiver + été)")
    print("   ✅ Rentrée scolaire")
    print("   ✅ Vacances scolaires")
    print("   ✅ Pic de Noël")
    print("\n🚀 Prochaines étapes :")
    print("   1. Vérifier : python verifier_base.py")
    print("   2. Prévisions : python forecast.py")
    print("=" * 80 + "\n")
# generate_data.py
"""
Générateur de données de test pour IA_Logistique_Durable
Génère des commandes et expéditions réalistes sur plusieurs mois
Usage: python generate_data.py
"""

import sqlite3
import random
from datetime import datetime, timedelta
from src.database import Database

class DataGenerator:
    def __init__(self, db_path='data/prevision.db'):
        self.db_path = db_path
        self.db = Database(db_path)

        # Récupérer les données de référence depuis la DB
        self.db.connect()

        # Clients (générés automatiquement si nécessaire)
        self.clients = ['CL001', 'CL002', 'CL003', 'CL004', 'CL005',
                        'CL006', 'CL007', 'CL008', 'CL009', 'CL010']

        # Produits (depuis Inventaire)
        result = self.db.query("SELECT ID_produit FROM Inventaire")
        self.produits = [row['ID_produit'] for row in result] if result else []

        # Transports (depuis Transport)
        result = self.db.query("SELECT ID_transport FROM Transport")
        self.transports = [row['ID_transport'] for row in result] if result else []

        self.db.disconnect()

        # Statuts
        self.statuts_commande = ['En attente', 'En cours', 'Livrée', 'Annulée']
        self.statuts_livraison = ['En transit', 'Livrée', 'Retardée', 'Annulée']

        # Prix moyens par produit
        self.prix_unitaires = {
            'P001': 5.5,  # Acier recyclé
            'P002': 2.8,  # Granulés plastique
            'P003': 2.8,  # LED
            'P004': 2.25,  # Cartons
            'P005': 12.5,  # Lubrifiant
            'P006': 6.2,  # Verre
            'P007': 2.8,  # Modules électroniques
            'P008': 0.675,  # Rubans
            'P009': 6.5,  # Colle
            'P010': 6.0  # Acier inoxydable
        }

        # Coûts transport par km
        self.cout_transport_km = {
            'TR001': 1.25, 'TR002': 0.65, 'TR003': 1.60, 'TR004': 0.45,
            'TR005': 0.25, 'TR006': 0.55, 'TR007': 1.10, 'TR008': 0.15
        }

        # Émissions CO2 par km
        self.emissions_transport_km = {
            'TR001': 0.90, 'TR002': 0.28, 'TR003': 0.08, 'TR004': 0.03,
            'TR005': 0.20, 'TR006': 0.20, 'TR007': 0.60, 'TR008': 0.01
        }

    def _get_next_id_commande(self):
        """Récupère le prochain ID de commande disponible"""
        self.db.connect()
        # Récupérer TOUS les IDs existants
        result = self.db.query("SELECT ID_commande FROM Commandes ORDER BY ID_commande DESC")
        self.db.disconnect()

        if not result:
            return 1

        # Extraire tous les numéros
        existing_ids = set()
        for row in result:
            try:
                num = int(row['ID_commande'].replace('CMD', ''))
                existing_ids.add(num)
            except:
                pass

        # Trouver le maximum + 1
        if existing_ids:
            return max(existing_ids) + 1
        return 1

    def _get_next_id_expedition(self):
        """Récupère le prochain ID d'expédition disponible"""
        self.db.connect()
        result = self.db.query("SELECT ID_expedition FROM Expeditions ORDER BY ID_expedition DESC")
        self.db.disconnect()

        if not result:
            return 1

        existing_ids = set()
        for row in result:
            try:
                num = int(row['ID_expedition'].replace('EXP', ''))
                existing_ids.add(num)
            except:
                pass

        if existing_ids:
            return max(existing_ids) + 1
        return 1

    def generer_commandes(self, date_debut, date_fin, nombre_commandes=None):
        """
        Génère des commandes entre deux dates avec saisonnalité réaliste

        Args:
            date_debut: date de début (str 'YYYY-MM-DD' ou datetime)
            date_fin: date de fin (str 'YYYY-MM-DD' ou datetime)
            nombre_commandes: nombre de commandes à générer (None = automatique)

        Returns:
            Liste des IDs de commandes générées
        """
        # Conversion des dates
        if isinstance(date_debut, str):
            date_debut = datetime.strptime(date_debut, '%Y-%m-%d')
        if isinstance(date_fin, str):
            date_fin = datetime.strptime(date_fin, '%Y-%m-%d')

        nombre_jours = (date_fin - date_debut).days

        if nombre_commandes is None:
            nombre_commandes = random.randint(int(nombre_jours * 3), int(nombre_jours * 5))

        print(f"\n{'=' * 70}")
        print(f"📦 GÉNÉRATION DE {nombre_commandes} COMMANDES")
        print(f"{'=' * 70}")
        print(f"📅 Période : {date_debut.date()} → {date_fin.date()}")
        print(f"📊 Durée : {nombre_jours} jours")

        # Récupérer le prochain ID disponible
        prochain_id = self._get_next_id_commande()
        print(f"🔢 Prochain ID disponible : CMD{prochain_id:03d}\n")

        self.db.connect()
        commandes_inserted = []

        for i in range(nombre_commandes):
            # Date aléatoire avec distribution réaliste
            jours_offset = random.randint(0, nombre_jours)
            date_commande = date_debut + timedelta(days=jours_offset)

            # Saisonnalité (plus d'activité en hiver et moins en été)
            mois = date_commande.month
            if mois in [11, 12, 1, 2]:  # Hiver - haute saison
                multiplicateur_saison = random.uniform(1.2, 1.5)
            elif mois in [6, 7, 8]:  # Été - basse saison
                multiplicateur_saison = random.uniform(0.7, 0.9)
            else:  # Printemps/Automne
                multiplicateur_saison = random.uniform(0.95, 1.15)

            # Effet jour de la semaine (moins le weekend)
            jour_semaine = date_commande.weekday()
            if jour_semaine >= 5:  # Weekend
                multiplicateur_jour = random.uniform(0.3, 0.6)
            else:  # Semaine
                multiplicateur_jour = random.uniform(0.9, 1.1)

            # ID commande unique
            id_commande = f"CMD{(prochain_id + i):03d}"
            id_client = random.choice(self.clients)
            id_produit = random.choice(self.produits)

            # Quantités de base par produit
            quantites_base = {
                'P001': (500, 5000),  # Acier
                'P002': (500, 4000),  # Plastique
                'P003': (200, 1500),  # LED
                'P004': (1000, 6000),  # Cartons
                'P005': (100, 800),  # Lubrifiant
                'P006': (500, 3000),  # Verre
                'P007': (200, 1000),  # Modules
                'P008': (500, 2500),  # Rubans
                'P009': (100, 600),  # Colle
                'P010': (1000, 5000)  # Acier inox
            }

            qte_min, qte_max = quantites_base.get(id_produit, (100, 1000))
            quantite_base = random.randint(qte_min, qte_max)
            quantite = int(quantite_base * multiplicateur_saison * multiplicateur_jour)
            quantite = max(quantite, 50)  # Minimum 50 unités

            # Calcul du coût
            prix_unitaire = self.prix_unitaires.get(id_produit, 5.0)
            cout_total = round(quantite * prix_unitaire, 2)

            # Délai de livraison (7 à 14 jours, parfois plus)
            delai = random.choices(
                [7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30],
                weights=[10, 15, 20, 20, 15, 10, 5, 3, 1, 0.5, 0.5]
            )[0]
            date_livraison = date_commande + timedelta(days=delai)

            # Statut basé sur l'ancienneté de la commande
            jours_depuis_commande = (datetime.now() - date_commande).days

            if jours_depuis_commande < 0:  # Commande future
                statut = 'En attente'
            elif jours_depuis_commande > 20:  # Ancienne
                statut = random.choices(
                    self.statuts_commande,
                    weights=[2, 5, 90, 3]  # Majorité livrée
                )[0]
            elif jours_depuis_commande > 10:  # Moyenne
                statut = random.choices(
                    self.statuts_commande,
                    weights=[5, 40, 50, 5]  # Mix en cours/livrée
                )[0]
            else:  # Récente
                statut = random.choices(
                    self.statuts_commande,
                    weights=[40, 50, 8, 2]  # Majorité en attente/cours
                )[0]

            commande = (
                id_commande,
                id_client,
                date_commande.strftime('%Y-%m-%d'),
                date_livraison.strftime('%Y-%m-%d'),
                id_produit,
                quantite,
                statut,
                cout_total
            )

            # Insertion immédiate pour éviter les conflits
            try:
                sql = """
                      INSERT INTO Commandes
                      (ID_commande, ID_client, Date_commande, Date_livraison_souhaitee,
                       ID_produit, Quantite_commandee, Statut_commande, Cout_total)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
                      """
                self.db.execute(sql, commande)
                commandes_inserted.append(id_commande)

                # Afficher progression tous les 100
                if (i + 1) % 100 == 0:
                    print(f"   ✓ {i + 1}/{nombre_commandes} commandes insérées...")

            except sqlite3.IntegrityError as e:
                print(f"   ⚠️  {id_commande} existe déjà, ignoré")

        print(f"\n✅ {len(commandes_inserted)} nouvelles commandes insérées !")
        self.db.disconnect()

        return commandes_inserted

    def generer_expeditions(self, id_commandes=None):
        """
        Génère des expéditions pour les commandes spécifiées

        Args:
            id_commandes: liste des ID de commandes (None = toutes sans expédition)

        Returns:
            Liste des IDs d'expéditions générées
        """
        self.db.connect()

        # Récupérer les commandes sans expédition
        if id_commandes is None or len(id_commandes) == 0:
            sql = """
                  SELECT c.ID_commande, \
                         c.Date_commande, \
                         c.Date_livraison_souhaitee,
                         c.Statut_commande, \
                         c.Quantite_commandee
                  FROM Commandes c
                           LEFT JOIN Expeditions e ON c.ID_commande = e.ID_commande
                  WHERE e.ID_expedition IS NULL
                    AND c.Statut_commande IN ('En cours', 'Livrée') \
                  """
            commandes = self.db.query(sql)
        else:
            placeholders = ','.join(['?' for _ in id_commandes])
            sql = f"""
            SELECT ID_commande, Date_commande, Date_livraison_souhaitee, 
                   Statut_commande, Quantite_commandee
            FROM Commandes
            WHERE ID_commande IN ({placeholders})
            AND Statut_commande IN ('En cours', 'Livrée')
            """
            commandes = self.db.query(sql, tuple(id_commandes))

        if not commandes:
            print("\n⚠️  Aucune commande éligible pour expédition")
            print(f"   (Statuts requis : 'En cours' ou 'Livrée')")
            self.db.disconnect()
            return []

        print(f"\n{'=' * 70}")
        print(f"🚚 GÉNÉRATION DE {len(commandes)} EXPÉDITIONS")
        print(f"{'=' * 70}\n")

        # Récupérer le prochain ID disponible
        prochain_id = self._get_next_id_expedition()
        print(f"🔢 Prochain ID disponible : EXP{prochain_id:03d}\n")

        expeditions_inserted = []

        for i, cmd in enumerate(commandes):
            id_expedition = f"EXP{(prochain_id + i):03d}"
            id_commande = cmd['ID_commande']

            # Choix du transport basé sur la quantité
            quantite = cmd.get('Quantite_commandee', 1000)
            if quantite > 10000:
                id_transport = random.choice(['TR001', 'TR004', 'TR005'])  # Gros transport
            elif quantite > 2000:
                id_transport = random.choice(['TR001', 'TR003', 'TR007'])  # Moyen
            else:
                id_transport = random.choice(['TR002', 'TR006', 'TR008'])  # Petit

            # Date d'expédition : 1-3 jours après la commande
            date_cmd = datetime.strptime(cmd['Date_commande'], '%Y-%m-%d')
            date_expedition = date_cmd + timedelta(days=random.randint(1, 3))

            # Date de livraison estimée
            date_livraison_souhaitee = datetime.strptime(cmd['Date_livraison_souhaitee'], '%Y-%m-%d')
            variation = random.randint(-2, 3)  # Peut être en avance ou en retard
            date_livraison_estimee = date_livraison_souhaitee + timedelta(days=variation)

            # Statut de livraison
            if cmd['Statut_commande'] == 'Livrée':
                statut_livraison = random.choices(
                    ['Livrée', 'Retardée'],
                    weights=[90, 10]
                )[0]
            else:  # En cours
                statut_livraison = random.choices(
                    ['En transit', 'Retardée'],
                    weights=[85, 15]
                )[0]

            # Distance réaliste (50 à 1500 km)
            distance = round(random.uniform(50, 1500), 2)

            # Coût transport
            cout_km = self.cout_transport_km.get(id_transport, 1.0)
            cout_transport = round(distance * cout_km, 2)

            # Émissions CO2
            emissions_km = self.emissions_transport_km.get(id_transport, 0.5)
            emissions = round(distance * emissions_km, 2)

            expedition = (
                id_expedition,
                id_commande,
                id_transport,
                date_expedition.strftime('%Y-%m-%d'),
                date_livraison_estimee.strftime('%Y-%m-%d'),
                statut_livraison,
                distance,
                cout_transport,
                emissions
            )

            # Insertion
            try:
                sql = """
                      INSERT INTO Expeditions
                      (ID_expedition, ID_commande, ID_transport, Date_expedition,
                       Date_livraison_estimee, Statut_livraison, Distance_parcourue,
                       Cout_transport, Emissions_CO2_effectives)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
                      """
                self.db.execute(sql, expedition)
                expeditions_inserted.append(id_expedition)

                # Afficher progression tous les 100
                if (i + 1) % 100 == 0:
                    print(f"   ✓ {i + 1}/{len(commandes)} expéditions insérées...")

            except sqlite3.IntegrityError:
                print(f"   ⚠️  {id_expedition} existe déjà, ignoré")

        print(f"\n✅ {len(expeditions_inserted)} nouvelles expéditions insérées !")
        self.db.disconnect()

        return expeditions_inserted

    def afficher_statistiques(self):
        """Affiche les statistiques complètes de la base de données"""
        self.db.connect()

        print(f"\n{'=' * 70}")
        print("📊 STATISTIQUES DE LA BASE DE DONNÉES")
        print(f"{'=' * 70}\n")

        # Commandes
        result = self.db.query("SELECT COUNT(*) as total FROM Commandes")
        total_commandes = result[0]['total'] if result else 0
        print(f"📦 Total commandes : {total_commandes}")

        if total_commandes > 0:
            # Par statut
            result = self.db.query("""
                                   SELECT Statut_commande, COUNT(*) as nb
                                   FROM Commandes
                                   GROUP BY Statut_commande
                                   ORDER BY nb DESC
                                   """)
            print("\n   Répartition par statut :")
            for row in result:
                pct = (row['nb'] / total_commandes) * 100
                print(f"      • {row['Statut_commande']:<15}: {row['nb']:>5} ({pct:>5.1f}%)")

            # Par période
            result = self.db.query("""
                                   SELECT strftime('%Y-%m', Date_commande) as mois, COUNT(*) as nb
                                   FROM Commandes
                                   GROUP BY mois
                                   ORDER BY mois DESC LIMIT 12
                                   """)
            if result:
                print("\n   Derniers 12 mois :")
                for row in result:
                    print(f"      • {row['mois']}: {row['nb']:>5} commandes")

            # Montant total
            result = self.db.query("SELECT SUM(Cout_total) as total FROM Commandes")
            if result and result[0]['total']:
                print(f"\n   💰 Chiffre d'affaires total : {result[0]['total']:,.2f} €")

        # Expéditions
        result = self.db.query("SELECT COUNT(*) as total FROM Expeditions")
        total_expeditions = result[0]['total'] if result else 0
        print(f"\n🚚 Total expéditions : {total_expeditions}")

        if total_expeditions > 0:
            # Par statut
            result = self.db.query("""
                                   SELECT Statut_livraison, COUNT(*) as nb
                                   FROM Expeditions
                                   GROUP BY Statut_livraison
                                   ORDER BY nb DESC
                                   """)
            print("\n   Répartition par statut :")
            for row in result:
                pct = (row['nb'] / total_expeditions) * 100
                print(f"      • {row['Statut_livraison']:<15}: {row['nb']:>5} ({pct:>5.1f}%)")

            # Émissions CO2
            result = self.db.query("SELECT SUM(Emissions_CO2_effectives) as total FROM Expeditions")
            if result and result[0]['total']:
                print(f"\n   🌍 Émissions CO2 totales : {result[0]['total']:,.2f} kg")

        # Commandes sans expédition
        result = self.db.query("""
                               SELECT COUNT(*) as nb
                               FROM Commandes c
                                        LEFT JOIN Expeditions e ON c.ID_commande = e.ID_commande
                               WHERE e.ID_expedition IS NULL
                               """)
        if result:
            nb_sans_exp = result[0]['nb']
            if nb_sans_exp > 0:
                print(f"\n⚠️  Commandes sans expédition : {nb_sans_exp}")

        print(f"\n{'=' * 70}\n")

        self.db.disconnect()


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   🚀 GÉNÉRATEUR DE DONNÉES - IA LOGISTIQUE DURABLE")
    print("=" * 70)

    generator = DataGenerator()

    # Vérifier que les tables existent
    if not generator.produits or not generator.transports:
        print("\n❌ ERREUR : Tables Inventaire ou Transport vides !")
        print("   Exécutez d'abord : python setup_database.py")
        exit(1)

    # CONFIGURATION
    print("\n📋 CONFIGURATION")
    print("-" * 70)

    # Générer des données sur 12 mois
    date_debut = '2024-01-01'
    date_fin = '2024-12-31'
    nombre_commandes = 1000  # ~3 commandes par jour

    print(f"📅 Période : {date_debut} → {date_fin}")
    print(f"📦 Nombre de commandes : {nombre_commandes}")
    print(f"📊 Produits disponibles : {len(generator.produits)}")
    print(f"🚚 Transports disponibles : {len(generator.transports)}")

    reponse = input("\n➡️  Générer les données ? (o/n) : ").lower()

    if reponse == 'o':
        print("\n🔄 Génération en cours...")

        # Génération des commandes
        commandes_ids = generator.generer_commandes(date_debut, date_fin, nombre_commandes)

        # Génération des expéditions pour les commandes créées
        if commandes_ids:
            generator.generer_expeditions(commandes_ids)

        # Afficher les statistiques
        generator.afficher_statistiques()

        print("\n" + "=" * 70)
        print("   ✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        print("\n🚀 Prochaines étapes :")
        print("   1. Vérifiez les données : python -m src.preprocessing")
        print("   2. Lancez les prévisions : python -m src.forecast")
        print()
    else:
        print("\n❌ Génération annulée")

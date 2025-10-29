from src.database import Database, afficher_tableau, afficher_statistiques


def exemple_consultation():
    """Exemples de consultation de données"""
    print("\n" + "=" * 80)
    print("🔍 EXEMPLES DE CONSULTATION DE LA BASE DE DONNÉES")

    # Connexion à la base
    db = Database("data/prévision.db")

    try:
        # Liste de toutes les tables disponibles
        print("\n📋 TABLES DISPONIBLES DANS LA BASE :")
        tables = db.query("SELECT name FROM sqlite_master WHERE type='table'")
        for table in tables:
            print(f"   - {table['name']}")

        # Exemple : afficher le tableau 'Fournisseurs'
        print("\n" + "-" * 50)
        afficher_tableau(db, "Fournisseurs")

        # Exemple : afficher le tableau 'Demande'
        print("\n" + "-" * 50)
        afficher_tableau(db, "Demande")

        # Exemple : statistiques détaillées
        print("\n" + "-" * 50)
        print("📊 STATISTIQUES DÉTAILLÉES")
        afficher_statistiques(db, "Fournisseurs")
        afficher_statistiques(db, "Demande")

        # Exemples de requêtes métier supplémentaires
        print("\n" + "-" * 50)
        print("🚀 REQUÊTES MÉTIER SPÉCIFIQUES")

        # Exemple 1: Fournisseurs avec leur capacité totale
        print("\n🏭 CAPACITÉ DES FOURNISSEURS :")
        capacite_fournisseurs = db.query("""
                                         SELECT nom, capacite_max, ville
                                         FROM Fournisseurs
                                         ORDER BY capacite_max DESC
                                         """)
        if capacite_fournisseurs:
            print("Fournisseurs par capacité :")
            for fournisseur in capacite_fournisseurs:
                print(f"   - {fournisseur['nom']}: {fournisseur['capacite_max']} unités ({fournisseur['ville']})")

        # Exemple 2: Demande moyenne
        print("\n📈 ANALYSE DE LA DEMANDE :")
        demande_stats = db.query("""
                                 SELECT AVG(quantite) as demande_moyenne,
                                        MAX(quantite) as demande_max,
                                        MIN(quantite) as demande_min
                                 FROM Demande
                                 """)
        if demande_stats:
            stats = demande_stats[0]
            print(f"   📊 Demande moyenne: {stats['demande_moyenne']:.2f}")
            print(f"   📈 Demande maximale: {stats['demande_max']}")
            print(f"   📉 Demande minimale: {stats['demande_min']}")

        # Exemple 3: Jointure entre Fournisseurs et Demande (si relation existe)
        print("\n🔗 RELATIONS FOURNISSEURS-DEMANDE :")
        try:
            relations = db.query("""
                                 SELECT f.nom as fournisseur, d.produit, d.quantite
                                 FROM Fournisseurs f
                                          JOIN Demande d ON f.id = d.fournisseur_id LIMIT 5
                                 """)
            if relations:
                print("Dernières relations trouvées :")
                for relation in relations:
                    print(f"   - {relation['fournisseur']} → {relation['produit']}: {relation['quantite']} unités")
            else:
                print("   ℹ️ Aucune relation directe trouvée entre les tables")
        except:
            print("   ℹ️ Structure de jointure non disponible")

    except Exception as e:
        print(f"❌ Erreur lors de la consultation : {e}")
    finally:
        # Toujours fermer la connexion
        db.disconnect()


def exemple_insertion_modification():
    """Exemples d'insertion et modification de données"""
    print("\n" + "=" * 80)
    print("✏️ EXEMPLES D'INSERTION ET MODIFICATION")

    db = Database("data/prévision.db")

    try:
        # Exemple d'insertion d'un nouveau fournisseur
        print("\n➕ AJOUT D'UN NOUVEAU FOURNISSEUR :")
        result = db.execute("""
                            INSERT INTO Fournisseurs (nom, capacite_max, ville, contact)
                            VALUES (?, ?, ?, ?)
                            """, ("Logistique Express", 5000, "Lyon", "contact@express.fr"))

        if result > 0:
            print("✅ Nouveau fournisseur ajouté avec succès !")
            afficher_tableau(db, "Fournisseurs")

        # Exemple de mise à jour
        print("\n🔄 MISE À JOUR DE CAPACITÉ :")
        result = db.execute("""
                            UPDATE Fournisseurs
                            SET capacite_max = capacite_max + 1000
                            WHERE ville = 'Lyon'
                            """)
        print(f"✅ {result} fournisseur(s) mis à jour")

    except Exception as e:
        print(f"❌ Erreur lors des modifications : {e}")
    finally:
        db.disconnect()


if __name__ == "__main__":
    exemple_consultation()
    exemple_insertion_modification()

    print("\n" + "=" * 80)
    print("🎯 EXÉCUTION TERMINÉE AVEC SUCCÈS !")
    print("=" * 80)
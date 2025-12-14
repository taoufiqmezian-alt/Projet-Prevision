"""
Script de diagnostic pour évaluer la qualité des prévisions
Vérifie les données, teste les modèles, génère un rapport de fiabilité
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

DB_PATH = "data/prevision.db"
REPORT_DIR = "diagnostic_report"
os.makedirs(REPORT_DIR, exist_ok=True)


# ====================================
# 1. DIAGNOSTIC DES DONNÉES
# ====================================

def diagnostic_base_donnees():
    """Analyse la qualité des données historiques"""

    print("=" * 80)
    print("📊 DIAGNOSTIC DE LA BASE DE DONNÉES")
    print("=" * 80)

    try:
        conn = sqlite3.connect(DB_PATH)

        # Vérifier les tables
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )
        print(f"\n✅ Tables trouvées : {tables['name'].tolist()}")

        # Analyser historique_commandes
        if 'historique_commandes' in tables['name'].values:
            df = pd.read_sql_query("SELECT * FROM historique_commandes", conn)

            print(f"\n📦 HISTORIQUE COMMANDES")
            print(f"   • Nombre d'enregistrements : {len(df):,}")
            print(f"   • Colonnes : {df.columns.tolist()}")
            print(f"   • Taille mémoire : {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

            # Période couverte
            if 'Date_commande' in df.columns:
                df['Date_commande'] = pd.to_datetime(df['Date_commande'], errors='coerce')
                date_min = df['Date_commande'].min()
                date_max = df['Date_commande'].max()
                duree_jours = (date_max - date_min).days
                duree_mois = duree_jours / 30.44

                print(f"\n📅 PÉRIODE COUVERTE")
                print(f"   • Première commande : {date_min.strftime('%Y-%m-%d')}")
                print(f"   • Dernière commande : {date_max.strftime('%Y-%m-%d')}")
                print(f"   • Durée totale : {duree_jours} jours ({duree_mois:.1f} mois)")

                # Évaluation durée
                if duree_mois < 6:
                    print(f"   ⚠️ ATTENTION : Moins de 6 mois de données (faible fiabilité)")
                elif duree_mois < 12:
                    print(f"   ⚠️ AVERTISSEMENT : 6-12 mois (fiabilité moyenne)")
                else:
                    print(f"   ✅ BON : {duree_mois:.1f} mois de données (bonne fiabilité)")

            # Analyse produits
            if 'ID_produit' in df.columns:
                nb_produits = df['ID_produit'].nunique()
                print(f"\n🏷️ PRODUITS")
                print(f"   • Nombre de produits uniques : {nb_produits}")

                # Distribution par produit
                dist_produits = df.groupby('ID_produit').size().describe()
                print(f"   • Commandes par produit (moyenne) : {dist_produits['mean']:.1f}")
                print(f"   • Commandes par produit (médiane) : {dist_produits['50%']:.1f}")
                print(f"   • Commandes par produit (min) : {int(dist_produits['min'])}")
                print(f"   • Commandes par produit (max) : {int(dist_produits['max'])}")

                # Produits avec peu de données
                points_par_produit = df.groupby('ID_produit').size()
                produits_faibles = points_par_produit[points_par_produit < 10]
                if len(produits_faibles) > 0:
                    print(f"   ⚠️ {len(produits_faibles)} produits avec <10 commandes (prévisions peu fiables)")

            # Qualité des données
            if 'Quantite' in df.columns:
                print(f"\n📈 QUALITÉ DES QUANTITÉS")
                print(f"   • Valeurs manquantes : {df['Quantite'].isnull().sum()}")
                print(f"   • Valeurs négatives : {(df['Quantite'] < 0).sum()}")
                print(f"   • Valeurs nulles : {(df['Quantite'] == 0).sum()}")
                print(f"   • Moyenne : {df['Quantite'].mean():.2f}")
                print(f"   • Médiane : {df['Quantite'].median():.2f}")
                print(f"   • Écart-type : {df['Quantite'].std():.2f}")

        # Analyser produits
        if 'produits' in tables['name'].values:
            df_prod = pd.read_sql_query("SELECT * FROM produits", conn)
            print(f"\n🏪 TABLE PRODUITS")
            print(f"   • Nombre de produits : {len(df_prod)}")
            print(f"   • Colonnes : {df_prod.columns.tolist()}")

            if 'Cout_par_unite' in df_prod.columns:
                print(f"   • Prix moyen : {df_prod['Cout_par_unite'].mean():.2f} €")
                print(f"   • Prix médian : {df_prod['Cout_par_unite'].median():.2f} €")

        conn.close()

        # Résumé qualité globale
        print(f"\n{'=' * 80}")
        print("🎯 ÉVALUATION GLOBALE DE LA QUALITÉ")
        print("=" * 80)

        score_qualite = 0
        criteres = []

        if duree_mois >= 12:
            score_qualite += 30
            criteres.append("✅ Durée suffisante (≥12 mois)")
        elif duree_mois >= 6:
            score_qualite += 15
            criteres.append("⚠️ Durée moyenne (6-12 mois)")
        else:
            criteres.append("❌ Durée insuffisante (<6 mois)")

        if nb_produits >= 5:
            score_qualite += 20
            criteres.append("✅ Diversité produits")

        if dist_produits['mean'] >= 20:
            score_qualite += 25
            criteres.append("✅ Historique riche par produit")
        elif dist_produits['mean'] >= 10:
            score_qualite += 15
            criteres.append("⚠️ Historique moyen par produit")
        else:
            criteres.append("❌ Historique faible par produit")

        taux_manquants = df['Quantite'].isnull().sum() / len(df)
        if taux_manquants < 0.01:
            score_qualite += 25
            criteres.append("✅ Données complètes")
        elif taux_manquants < 0.05:
            score_qualite += 15
            criteres.append("⚠️ Quelques données manquantes")
        else:
            criteres.append("❌ Beaucoup de données manquantes")

        for critere in criteres:
            print(f"   {critere}")

        print(f"\n📊 Score de qualité : {score_qualite}/100")

        if score_qualite >= 80:
            print("   ✅ EXCELLENTE base de données - Prévisions fiables")
        elif score_qualite >= 60:
            print("   ⚠️ BONNE base de données - Prévisions utilisables avec précautions")
        elif score_qualite >= 40:
            print("   ⚠️ BASE MOYENNE - Prévisions à valider manuellement")
        else:
            print("   ❌ BASE INSUFFISANTE - Collecter plus de données avant prévisions")

        return df, score_qualite

    except Exception as e:
        print(f"❌ ERREUR : {e}")
        return None, 0


# ====================================
# 2. TEST DE BACKTESTING
# ====================================

def backtesting_simple(df, test_days=30):
    """Teste la précision des prévisions sur le passé récent"""

    print(f"\n{'=' * 80}")
    print("🔮 TEST DE BACKTESTING (Validation Historique)")
    print("=" * 80)
    print(f"Stratégie : Utiliser tout sauf les {test_days} derniers jours pour prédire ces {test_days} jours")

    try:
        from xgboost import XGBRegressor

        df['Date_commande'] = pd.to_datetime(df['Date_commande'])
        date_split = df['Date_commande'].max() - timedelta(days=test_days)

        resultats = []

        for prod_id in df['ID_produit'].unique():
            df_prod = df[df['ID_produit'] == prod_id].copy()

            if len(df_prod) < 20:
                continue

            # Split train/test
            train = df_prod[df_prod['Date_commande'] < date_split].copy()
            test = df_prod[df_prod['Date_commande'] >= date_split].copy()

            if len(train) < 10 or len(test) < 5:
                continue

            # Préparer features
            train['Jours'] = (train['Date_commande'] - train['Date_commande'].min()).dt.days
            test['Jours'] = (test['Date_commande'] - train['Date_commande'].min()).dt.days

            X_train = train[['Jours']].values
            y_train = train['Quantite'].values
            X_test = test[['Jours']].values
            y_test = test['Quantite'].values

            # Modèle simple
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Prédire
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)

            # Métriques
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))

            resultats.append({
                'Produit': prod_id,
                'MAPE': mape,
                'RMSE': rmse,
                'MAE': mae,
                'Nb_test': len(y_test)
            })

            print(f"\n   Produit {prod_id}:")
            print(f"      • MAPE: {mape:.2f}%")
            print(f"      • RMSE: {rmse:.2f}")
            print(f"      • MAE:  {mae:.2f}")

        if resultats:
            df_resultats = pd.DataFrame(resultats)

            print(f"\n{'=' * 80}")
            print("📊 RÉSULTATS MOYENS DU BACKTESTING")
            print("=" * 80)
            print(f"   • MAPE moyen : {df_resultats['MAPE'].mean():.2f}%")
            print(f"   • RMSE moyen : {df_resultats['RMSE'].mean():.2f}")
            print(f"   • MAE moyen  : {df_resultats['MAE'].mean():.2f}")

            # Interprétation
            mape_moyen = df_resultats['MAPE'].mean()
            print(f"\n🎯 INTERPRÉTATION :")
            if mape_moyen < 10:
                print("   ✅ EXCELLENTE précision (<10% d'erreur)")
                print("   → Vos prévisions sont TRÈS FIABLES")
            elif mape_moyen < 20:
                print("   ✅ BONNE précision (10-20% d'erreur)")
                print("   → Vos prévisions sont FIABLES pour la planification")
            elif mape_moyen < 30:
                print("   ⚠️ PRÉCISION MOYENNE (20-30% d'erreur)")
                print("   → Utilisez les prévisions avec une marge de sécurité")
            else:
                print("   ❌ PRÉCISION FAIBLE (>30% d'erreur)")
                print("   → Améliorez le modèle ou collectez plus de données")

            # Sauvegarder
            df_resultats.to_csv(f"{REPORT_DIR}/backtesting_resultats.csv", index=False)

            # Graphique
            plt.figure(figsize=(10, 6))
            plt.bar(df_resultats['Produit'].astype(str), df_resultats['MAPE'])
            plt.axhline(y=20, color='orange', linestyle='--', label='Seuil acceptable (20%)')
            plt.axhline(y=10, color='green', linestyle='--', label='Seuil excellent (10%)')
            plt.xlabel('Produit')
            plt.ylabel('MAPE (%)')
            plt.title('Erreur de Prévision par Produit (Backtesting)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{REPORT_DIR}/backtesting_mape.png", dpi=100)
            plt.close()

            return df_resultats
        else:
            print("⚠️ Pas assez de données pour le backtesting")
            return None

    except Exception as e:
        print(f"❌ Erreur backtesting : {e}")
        return None


# ====================================
# 3. VISUALISATION TENDANCES
# ====================================

def analyser_tendances(df):
    """Visualise les tendances historiques"""

    print(f"\n{'=' * 80}")
    print("📈 ANALYSE DES TENDANCES")
    print("=" * 80)

    try:
        df['Date_commande'] = pd.to_datetime(df['Date_commande'])

        for prod_id in df['ID_produit'].unique()[:5]:  # Top 5 produits
            df_prod = df[df['ID_produit'] == prod_id].copy()
            df_prod = df_prod.sort_values('Date_commande')

            plt.figure(figsize=(12, 5))

            # Graphique 1 : Quantités
            plt.subplot(1, 2, 1)
            plt.plot(df_prod['Date_commande'], df_prod['Quantite'], marker='o', alpha=0.6)
            plt.title(f'Produit {prod_id} - Quantités')
            plt.xlabel('Date')
            plt.ylabel('Quantité')
            plt.grid(True, alpha=0.3)

            # Graphique 2 : Moyenne mobile 7 jours
            plt.subplot(1, 2, 2)
            df_prod['MA7'] = df_prod['Quantite'].rolling(window=7, min_periods=1).mean()
            plt.plot(df_prod['Date_commande'], df_prod['Quantite'],
                     marker='o', alpha=0.3, label='Quantité réelle')
            plt.plot(df_prod['Date_commande'], df_prod['MA7'],
                     color='red', linewidth=2, label='Moyenne mobile 7j')
            plt.title(f'Produit {prod_id} - Lissage')
            plt.xlabel('Date')
            plt.ylabel('Quantité')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{REPORT_DIR}/tendance_produit_{prod_id}.png", dpi=100)
            plt.close()

            print(f"   ✅ Graphique créé pour produit {prod_id}")

        print(f"\n✅ Graphiques sauvegardés dans '{REPORT_DIR}/'")

    except Exception as e:
        print(f"❌ Erreur visualisation : {e}")


# ====================================
# 4. RAPPORT FINAL
# ====================================

def generer_rapport_final(score_qualite, resultats_backtest):
    """Génère un rapport de synthèse"""

    print(f"\n{'=' * 80}")
    print("📄 RAPPORT FINAL - VOS PRÉVISIONS SONT-ELLES RÉELLES ?")
    print("=" * 80)

    rapport = []
    score_final = 0

    # Critère 1 : Qualité données
    rapport.append(f"\n1️⃣ QUALITÉ DES DONNÉES : {score_qualite}/100")
    if score_qualite >= 80:
        rapport.append("   ✅ Excellente base de données")
        score_final += 40
    elif score_qualite >= 60:
        rapport.append("   ⚠️ Bonne base mais améliorable")
        score_final += 25
    else:
        rapport.append("   ❌ Base insuffisante")
        score_final += 10

    # Critère 2 : Performance backtesting
    if resultats_backtest is not None and len(resultats_backtest) > 0:
        mape_moyen = resultats_backtest['MAPE'].mean()
        rapport.append(f"\n2️⃣ PRÉCISION DES PRÉVISIONS : MAPE {mape_moyen:.2f}%")
        if mape_moyen < 15:
            rapport.append("   ✅ Excellente précision")
            score_final += 40
        elif mape_moyen < 25:
            rapport.append("   ⚠️ Bonne précision")
            score_final += 25
        else:
            rapport.append("   ❌ Précision insuffisante")
            score_final += 10
    else:
        rapport.append("\n2️⃣ PRÉCISION : Non testée")
        score_final += 10

    # Critère 3 : Cohérence
    rapport.append(f"\n3️⃣ COHÉRENCE MÉTIER :")
    rapport.append("   ⚠️ À valider manuellement avec votre équipe")
    score_final += 10  # Score neutre

    # Verdict final
    rapport.append(f"\n{'=' * 80}")
    rapport.append(f"🎯 SCORE FINAL DE FIABILITÉ : {score_final}/100")
    rapport.append("=" * 80)

    if score_final >= 80:
        rapport.append("\n✅ VOS PRÉVISIONS SONT FIABLES ET RÉELLES")
        rapport.append("   → Vous pouvez les utiliser pour la planification")
        rapport.append("   → Recommandation : Surveiller mensuellement")
    elif score_final >= 60:
        rapport.append("\n⚠️ VOS PRÉVISIONS SONT UTILISABLES AVEC PRÉCAUTIONS")
        rapport.append("   → Ajoutez une marge de sécurité de 20-30%")
        rapport.append("   → Recommandation : Valider avec équipe logistique")
    else:
        rapport.append("\n❌ VOS PRÉVISIONS NE SONT PAS ENCORE FIABLES")
        rapport.append("   → Collectez plus de données (objectif : 12 mois)")
        rapport.append("   → Enrichissez avec variables externes")
        rapport.append("   → Recommandation : Continuer avec méthodes manuelles")

    for ligne in rapport:
        print(ligne)

    # Sauvegarder rapport
    with open(f"{REPORT_DIR}/rapport_diagnostic.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(rapport))

    print(f"\n📁 Rapport complet sauvegardé : {REPORT_DIR}/rapport_diagnostic.txt")


# ====================================
# SCRIPT PRINCIPAL
# ====================================

if __name__ == "__main__":
    print("\n🔍 LANCEMENT DU DIAGNOSTIC COMPLET")
    print("=" * 80)
    print("Ce script va :")
    print("  1. Analyser la qualité de vos données")
    print("  2. Tester la précision des prévisions (backtesting)")
    print("  3. Visualiser les tendances")
    print("  4. Générer un rapport de fiabilité")
    print("=" * 80)

    # Étape 1 : Diagnostic données
    df_history, score_qualite = diagnostic_base_donnees()

    if df_history is not None and len(df_history) > 0:
        # Étape 2 : Backtesting
        resultats_backtest = backtesting_simple(df_history, test_days=30)

        # Étape 3 : Tendances
        analyser_tendances(df_history)

        # Étape 4 : Rapport final
        generer_rapport_final(score_qualite, resultats_backtest)

        print(f"\n{'=' * 80}")
        print("✅ DIAGNOSTIC TERMINÉ")
        print(f"📁 Tous les fichiers sont dans : {REPORT_DIR}/")
        print("=" * 80)
    else:
        print("\n❌ Impossible de continuer sans données valides")
from src.preprocessing import DataPreprocessor, print_data_quality_report

def main():
    print("📌 Chargement des données depuis la base...")
    data_dict = DataPreprocessor.load_data_from_db()

    for table_name, df in data_dict.items():
        print(f"\n📂 Table: {table_name}")
        print(df.head())  # Affiche les 5 premières lignes
        print_data_quality_report(df, table_name)

if __name__ == "__main__":
    main()

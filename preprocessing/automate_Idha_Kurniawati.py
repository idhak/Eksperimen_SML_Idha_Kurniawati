# automate_Idha_Kurniawati.py

# Import libraries
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump

# Lokasi file
RAW_DATA_PATH = "data_raw/heart.csv"
OUTPUT_DIR = "preprocessing/data_preprocessing"

# Buat folder output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print(f"📥 Memuat data dari: {filepath}")
    return pd.read_csv(filepath)

# Fungsi untuk menyiapkan pipeline preprocessing
def get_preprocessing_pipeline():
    print("🔧 Menyiapkan pipeline preprocessing...")
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features

# Fungsi untuk menyimpan hasil preprocessing
def save_processed_outputs(X_train, X_test, y_train, y_test, feature_names, fitted_pipeline, output_dir):
    pd.DataFrame(X_train, columns=feature_names).to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    # Simpan nama fitur
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(name + "\n")

    # Simpan pipeline yang sudah fit
    dump(fitted_pipeline, os.path.join(output_dir, 'preprocessor_pipeline.joblib'))
    print("✅ Semua file output berhasil disimpan.")

# Fungsi utama untuk memuat, memproses, dan menyimpan data
def preprocess_and_save(input_path, output_dir):
    df = load_data(input_path)
    if 'HeartDisease' not in df.columns:
        raise ValueError("Kolom 'HeartDisease' tidak ditemukan dalam dataset!")

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    pipeline, numeric_features, categorical_features = get_preprocessing_pipeline()
    print("⚙️  Melakukan preprocessing...")

    # Fit dan transform
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Ambil nama fitur setelah one-hot encoding
    cat_feature_names = pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    final_feature_names = numeric_features + list(cat_feature_names)

    # Simpan semua output termasuk pipeline yang sudah fit
    save_processed_outputs(X_train_transformed, X_test_transformed, y_train, y_test, final_feature_names, pipeline, output_dir)

    # Tampilkan ringkasan
    print("\n📌 Ringkasan Output:")
    print("- X_train shape:", X_train_transformed.shape)
    print("- X_test shape :", X_test_transformed.shape)
    print("- Jumlah fitur :", len(final_feature_names))
    
    print("\n🔍 Contoh Data:")
    print(pd.DataFrame(X_train_transformed, columns=final_feature_names).head())

# Main execution
if __name__ == "__main__":
    preprocess_and_save(RAW_DATA_PATH, OUTPUT_DIR)

# app.py (Lengkap: Prediksi | Evaluasi Model (K-Fold) | Dokumentasi)
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(layout="wide", page_title="Sistem Prediksi Fertilitas Pria")

# -------------------------
# Util: Load & prepare data
# -------------------------
@st.cache_data
def load_data(path="fertility.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "childish diseases": "childish_diseases",
        "accident or serious trauma": "accident_trauma",
        "surgical intervention": "surgical",
        "high fevers in the last year": "high_fever",
        "frequency of alcohol consumption": "alcohol",
        "smoking habit": "smoking",
        "number of hours spent sitting per day": "hours_sitting"
    })
    # normalize strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    # normalize diagnosis labels
    df["diagnosis"] = df["diagnosis"].replace({
        "n": "normal", "normal": "normal",
        "o": "altered", "altered": "altered"
    })
    df = df.dropna(subset=["diagnosis"])
    return df

@st.cache_data
def encode_dataset(df):
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == object:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    return df_encoded, label_encoders

df = load_data()
if df is None:
    st.title("🧬 Sistem Prediksi Fertilitas Pria")
    st.error("❌ File 'fertility.csv' tidak ditemukan. Upload file lalu refresh.")
    st.stop()

df_encoded, label_encoders = encode_dataset(df)

# -------------------------
# Sidebar menu
# -------------------------
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih halaman:", ["Prediksi", "Evaluasi Model", "Dokumentasi & Teori"])

# helper model prototypes
def get_model_prototypes():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True)
    }

def model_filename(name):
    return {
        "Random Forest": "model_rf.pkl",
        "KNN": "model_knn.pkl",
        "SVM": "model_svm.pkl"
    }[name]

# -------------------------
# PAGE: PREDIKSI
# -------------------------
if menu == "Prediksi":
    st.header("📝 Prediksi Fertilitas")

    mapping_options = {
        "season": {"Musim Semi":"spring","Musim Panas":"summer","Musim Gugur":"fall","Musim Dingin":"winter"},
        "childish_diseases": {"Ya":"yes","Tidak":"no"},
        "accident_trauma": {"Ya":"yes","Tidak":"no"},
        "surgical": {"Ya":"yes","Tidak":"no"},
        "high_fever": {"Kurang dari 3 bulan lalu":"less than 3 months ago","Lebih dari 3 bulan lalu":"more than 3 months ago","Tidak Pernah":"no"},
        "alcohol": {"Tidak Pernah":"hardly ever or never","Seminggu sekali":"once a week","Beberapa kali seminggu":"several times a week","Setiap hari":"every day"},
        "smoking": {"Tidak merokok":"never","Jarang":"occasional","Sering":"daily"}
    }

    # input fields
    user_input = {}
    for col in df.columns:
        if col == "diagnosis": continue
        if col in mapping_options:
            pilihan = st.selectbox(col.replace("_"," ").title(), list(mapping_options[col].keys()))
            user_input[col] = mapping_options[col][pilihan]
        else:
            if col=="age":
                user_input[col] = st.slider("Usia", 18, 60, 25)
            elif col=="hours_sitting":
                user_input[col] = st.slider("Jam Duduk/hari", 0, 16, 5)
            else:
                user_input[col] = st.number_input(col.replace("_"," ").title(), value=float(df[col].median()))

    # pilih metode
    metode = st.selectbox("Pilih Metode ML", ["Random Forest","KNN","SVM"])

    # tombol prediksi
    if st.button("Prediksi"):
        input_df = pd.DataFrame([user_input])
        filename = model_filename(metode)
        if not os.path.exists(filename):
            st.warning(f"File model '{filename}' tidak ditemukan. Latih dulu di halaman Evaluasi Model.")
        else:
            saved = joblib.load(filename)
            model = saved["model"]
            le_saved = saved["label_encoders"]
            scaler = saved["scaler"]
            cols = saved["columns"]

            for col in input_df.columns:
                if col in le_saved:
                    input_df[col] = le_saved[col].transform(input_df[col].astype(str).str.lower())
                else:
                    input_df[col] = pd.to_numeric(input_df[col])

            input_df = input_df[cols]
            input_scaled = scaler.transform(input_df)
            pred_encoded = model.predict(input_scaled)[0]
            pred_label = le_saved["diagnosis"].inverse_transform([pred_encoded])[0]

            st.markdown("### 🔍 Hasil Prediksi")
            if pred_label.lower() == "normal":
                st.success(f"🟢 Hasil ({metode}): Normal — tidak ada indikasi penurunan fertilitas.")
            else:
                st.error(f"⚠ Hasil ({metode}): Altered — ada indikasi penurunan fertilitas.")

# -------------------------
# PAGE: EVALUASI MODEL
# -------------------------
elif menu == "Evaluasi Model":
    st.header("📊 Evaluasi Model & Validasi (K-Fold)")

    st.markdown("Melatih model, menjalankan K-Fold, melihat confusion matrix & classification report.")

    cv_folds = st.sidebar.selectbox("Jumlah fold K-Fold:", [3,5,10], index=1)
    run_eval = st.button("Jalankan Evaluasi (Cross-Validation)")

    if st.button("Latih & Simpan Model (RF, KNN, SVM)"):
        st.info("Melatih model pada seluruh dataset ...")
        X = df_encoded.drop("diagnosis", axis=1)
        y = df_encoded["diagnosis"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        prototypes = get_model_prototypes()
        for name, m in prototypes.items():
            fname = model_filename(name)
            st.write(f"Melatih: {name} ...")
            m.fit(X_scaled, y)
            joblib.dump({
                "model": m,
                "label_encoders": label_encoders,
                "scaler": scaler,
                "columns": list(X.columns)
            }, fname)
            st.success(f"{name} disimpan sebagai {fname}")
        st.balloons()

    if run_eval:
        st.info(f"Menjalankan {cv_folds}-fold cross-validation ...")
        X = df_encoded.drop("diagnosis", axis=1)
        y = df_encoded["diagnosis"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        prototypes = get_model_prototypes()

        for name, model in prototypes.items():
            st.write(f"---\n**Model:** {name}")
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
            st.write(f"Akurasi per fold: {np.round(scores,4)} — rata-rata: {np.round(scores.mean(),4)}")

            y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
            cm = confusion_matrix(y, y_pred)
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(4,3))
            ax.imshow(cm, aspect='auto')
            ax.set_title(f"Confusion Matrix — {name}")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            labels = label_encoders["diagnosis"].classes_
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(labels); ax.set_yticklabels(labels)
            for (i,j), val in np.ndenumerate(cm):
                ax.text(j,i,val,ha='center',va='center')
            st.pyplot(fig)

            st.write("Classification Report:")
            clf_rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
            df_rep = pd.DataFrame(clf_rep).transpose().round(4)
            st.dataframe(df_rep)

        st.success("Evaluasi selesai.")

# -------------------------
# PAGE: DOKUMENTASI & TEORI
# -------------------------
elif menu == "Dokumentasi & Teori":
    st.header("📘 Dokumentasi & Teori Metode")

    st.markdown("""
    ### Ringkasan Sistem
    Aplikasi ini menggunakan dataset `fertility.csv` untuk memprediksi status fertilitas (Normal / Altered).
    Alur umum:
    1. Load dataset → preprocessing → label encoding.
    2. Fitur numerik discale saat training/prediksi.
    3. Model ML (Random Forest, KNN, SVM) dilatih.
    4. Input user di-encode & di-scale sebelum prediksi.
    """)

    st.markdown("### Metode yang dipakai")
    st.markdown("""
    **Random Forest** — ensemble berbasis decision tree. Stabil & baik untuk fitur kategorikal.  
    **KNN** — instance-based; efektif untuk dataset kecil.  
    **SVM** — mencari hyperplane optimal; baik untuk margin jelas.
    """)

    st.markdown("### Cara Penggunaan")
    st.markdown("""
    1. Pastikan `fertility.csv` ada di folder aplikasi.  
    2. Jika belum punya model `.pkl`, buka halaman Evaluasi Model → Latih & Simpan Model.  
    3. Buka Prediksi → isi input → pilih metode → klik Prediksi.  
    4. Untuk laporan, gunakan halaman Evaluasi (Confusion Matrix & Classification Report).
    """)

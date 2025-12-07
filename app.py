import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

st.title("🧬 Sistem Prediksi Fertilitas Pria (Versi Bahasa Indonesia)")

# ==========================================================
# LOAD DATASET
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("fertility.csv", sep=";")

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

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    df["diagnosis"] = df["diagnosis"].replace({
        "N": "Normal", "n": "Normal", "normal": "Normal", "N ": "Normal",
        "O": "Altered", "o": "Altered", "altered": "Altered", " O": "Altered"
    })

    df = df.dropna(subset=["diagnosis"])
    return df


df = load_data()

# ==========================================================
# LABEL ENCODING DATASET
# ==========================================================
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


df_encoded, label_encoders = encode_dataset(df)

# ==========================================================
# FORM INPUT BAHASA INDONESIA
# ==========================================================
st.subheader("📝 Form Prediksi")

user_input = {}

mapping_options = {
    "season": {
        "Musim Semi": "spring",
        "Musim Panas": "summer",
        "Musim Gugur": "fall",
        "Musim Dingin": "winter",
    },
    "childish_diseases": {"Ya": "yes", "Tidak": "no"},
    "accident_trauma": {"Ya": "yes", "Tidak": "no"},
    "surgical": {"Ya": "yes", "Tidak": "no"},
    "high_fever": {
        "Kurang dari 3 bulan lalu": "less than 3 months ago",
        "Lebih dari 3 bulan lalu": "more than 3 months ago",
        "Tidak Pernah": "no",
    },
    "alcohol": {
        "Tidak Pernah": "hardly ever or never",
        "Seminggu sekali": "once a week",
        "Beberapa kali seminggu": "several times a week",
        "Setiap hari": "every day",
    },
    "smoking": {
        "Tidak merokok": "never",
        "Jarang": "occasional",
        "Sering": "daily",
    }
}

for col in df.columns:
    if col == "diagnosis":
        continue

    if col in mapping_options:
        indo_choices = list(mapping_options[col].keys())
        pilihan = st.selectbox(col.replace("_", " ").title(), indo_choices)
        user_input[col] = mapping_options[col][pilihan]

    else:
        if col == "age":
            user_input[col] = st.slider("Usia", 18, 36, 25)
        elif col == "hours_sitting":
            user_input[col] = st.slider("Jam Duduk per Hari", 0, 16, 5)
        else:
            user_input[col] = df[col].median()

input_df = pd.DataFrame([user_input])

# ==========================================================
# PREDIKSI
# ==========================================================
st.subheader("🔍 Hasil Prediksi")

if st.button("Prediksi Risiko"):
    try:
        saved = joblib.load("model.pkl")
        model = saved["model"]
        label_encoders = saved["label_encoders"]
        scaler = saved["scaler"]
        cols = saved["columns"]

        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        input_scaled = scaler.transform(input_df[cols])

        pred = model.predict(input_scaled)[0]
        hasil = label_encoders["diagnosis"].inverse_transform([pred])[0]

        if hasil == "Normal":
            st.success("🟢 Hasil: Normal (Tidak ada indikasi penurunan fertilitas)")
        else:
            st.error("⚠ Altered / Risiko Tinggi (Ada indikasi penurunan fertilitas)")

    except FileNotFoundError:
        st.warning("⚠ Model belum tersedia. Silakan pastikan file *model.pkl* ada.")

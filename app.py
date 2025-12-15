import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Clustering & Regression - Online Retail",
    layout="wide"
)

st.title("📊 Clustering & Regression - Online Retail Dataset")
st.write("""
Aplikasi ini menyediakan:
- **Clustering pelanggan (K-Means)**
- **Regresi Linear**
- **Input data manual & visualisasi regresi**
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel("Online Retail.xlsx")

df = load_data()
st.subheader("Preview Dataset")
st.dataframe(df.head())

# =====================================================
# PREPROCESSING
# =====================================================
df = df.dropna(subset=['CustomerID'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# =====================================================
# RFM
# =====================================================
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
st.subheader("Data RFM")
st.dataframe(rfm.head())

# =====================================================
# NORMALISASI & CLUSTERING
# =====================================================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

k = st.slider("Pilih jumlah cluster (k)", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

sil = silhouette_score(rfm_scaled, rfm['Cluster'])
st.write("**Silhouette Score:**", round(sil, 3))

# =====================================================
# PCA VISUALISASI CLUSTER
# =====================================================
st.subheader("Visualisasi Cluster (PCA)")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(rfm_scaled)

fig1, ax1 = plt.subplots()
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=rfm['Cluster'],
    palette="Set1",
    ax=ax1
)
ax1.set_title("Visualisasi Cluster Pelanggan")
st.pyplot(fig1)

# =====================================================
# REGRESI LINEAR GLOBAL
# =====================================================
st.subheader("Regresi Linear Global")

X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.write("R² Score:", round(r2_score(y, y_pred), 3))
st.write("RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 2))

# =====================================================
# VISUALISASI REGRESI GLOBAL
# =====================================================
st.markdown("### Visualisasi Regresi Global")

fig2, ax2 = plt.subplots()
ax2.scatter(y, y_pred, alpha=0.6)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax2.set_xlabel("Nilai Aktual Monetary")
ax2.set_ylabel("Nilai Prediksi Monetary")
ax2.set_title("Aktual vs Prediksi (Global)")
st.pyplot(fig2)

# =====================================================
# INPUT DATA REGRESI (INI YANG DOSEN MAU)
# =====================================================
st.subheader("🔢 Input Data untuk Prediksi Regresi")

recency_input = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0)
frequency_input = st.number_input("Frequency (jumlah transaksi)", min_value=1)

if st.button("Prediksi Monetary Value"):
    input_data = np.array([[recency_input, frequency_input]])
    prediction = model.predict(input_data)
    
    st.success(f"💰 Prediksi Monetary Value: {prediction[0]:,.2f}")

# =====================================================
# VISUALISASI INPUT TERHADAP MODEL
# =====================================================
st.markdown("### Visualisasi Input terhadap Model")

fig3, ax3 = plt.subplots()
ax3.scatter(rfm['Frequency'], rfm['Monetary'], alpha=0.4, label="Data Asli")
ax3.scatter(frequency_input, prediction, color='red', s=100, label="Input User")
ax3.set_xlabel("Frequency")
ax3.set_ylabel("Monetary")
ax3.legend()
st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =====================================================
# JUDUL APLIKASI
# =====================================================
st.set_page_config(page_title="Clustering & Regression - Online Retail", layout="wide")

st.title("📊 Clustering & Regression - Online Retail Dataset")
st.write("""
Aplikasi ini melakukan:
1. Segmentasi pelanggan menggunakan **K-Means Clustering**
2. Analisis **Regresi Linear**
berdasarkan dataset **Online Retail**.
""")

# =====================================================
# LOAD DATASET
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel("Online Retail.xlsx")

df = load_data()

st.subheader("Preview Dataset")
st.dataframe(df.head())

# =====================================================
# PREPROCESSING DATA
# =====================================================
st.subheader("Preprocessing Data")

# Hapus data kosong
df = df.dropna(subset=['CustomerID'])

# Hitung total harga
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Konversi tanggal
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

st.write("Jumlah data setelah preprocessing:", df.shape[0])

# =====================================================
# MEMBENTUK DATA RFM
# =====================================================
st.subheader("Pembentukan Data RFM")

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

st.dataframe(rfm.head())

# =====================================================
# NORMALISASI DATA
# =====================================================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# =====================================================
# ELBOW METHOD
# =====================================================
st.subheader("Menentukan Jumlah Cluster (Elbow Method)")

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), wcss, marker='o')
ax1.set_xlabel("Jumlah Cluster (k)")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

# =====================================================
# CLUSTERING K-MEANS
# =====================================================
st.subheader("Proses Clustering")

k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

st.dataframe(rfm.head())

# =====================================================
# EVALUASI CLUSTER
# =====================================================
sil = silhouette_score(rfm_scaled, rfm['Cluster'])
st.write("**Silhouette Score:**", round(sil, 3))

# =====================================================
# VISUALISASI CLUSTER (PCA)
# =====================================================
st.subheader("Visualisasi Cluster (PCA)")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(rfm_scaled)

fig2, ax2 = plt.subplots()
sns.scatterplot(
    x=pca_data[:, 0],
    y=pca_data[:, 1],
    hue=rfm['Cluster'],
    palette='Set1',
    ax=ax2
)
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_title("Visualisasi Cluster Pelanggan")
st.pyplot(fig2)

# =====================================================
# INTERPRETASI CLUSTER
# =====================================================
st.subheader("Rata-rata Nilai RFM per Cluster")
cluster_summary = rfm.groupby('Cluster').mean()
st.dataframe(cluster_summary)

# =====================================================
# REGRESI LINEAR (GLOBAL)
# =====================================================
st.subheader("Regresi Linear Global")

X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

st.write("**R2 Score:**", round(r2_score(y, y_pred), 3))
st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y, y_pred)), 2))

# =====================================================
# REGRESI PER CLUSTER
# =====================================================
st.subheader("Regresi Linear per Cluster")

for c in sorted(rfm['Cluster'].unique()):
    st.write(f"### Cluster {c}")
    
    data_c = rfm[rfm['Cluster'] == c]
    Xc = data_c[['Recency', 'Frequency']]
    yc = data_c['Monetary']
    
    reg = LinearRegression()
    reg.fit(Xc, yc)
    ycp = reg.predict(Xc)
    
    st.write("R2 Score:", round(r2_score(yc, ycp), 3))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(yc, ycp)), 2))

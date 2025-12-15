import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Online Retail Clustering & Regression",
    layout="wide"
)

st.title("📊 Clustering & Regression - Online Retail Dataset")
st.write("""
Aplikasi ini menampilkan:
- Segmentasi pelanggan menggunakan **K-Means Clustering**
- Prediksi **Total Spending** menggunakan **Random Forest Regression**
- Input data manual oleh pengguna
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel("Online Retail.xlsx")

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# =====================================================
# FEATURE ENGINEERING (CUSTOMER LEVEL)
# =====================================================
customer_df = df.groupby("CustomerID").agg({
    "Quantity": "sum",
    "TotalPrice": "sum"
}).reset_index()

customer_df.columns = ["CustomerID", "TotalQuantity", "TotalSpending"]

st.subheader("📌 Data Pelanggan (Agregasi)")
st.dataframe(customer_df.head())

# =====================================================
# CLUSTERING - KMEANS
# =====================================================
st.subheader("🔹 Clustering Pelanggan (K-Means)")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    customer_df[["TotalQuantity", "TotalSpending"]]
)

k = st.slider("Jumlah Cluster (K)", min_value=2, max_value=6, value=3)

kmeans = KMeans(n_clusters=k, random_state=42)
customer_df["Cluster"] = kmeans.fit_predict(scaled_data)

fig1, ax1 = plt.subplots()
ax1.scatter(
    customer_df["TotalQuantity"],
    customer_df["TotalSpending"],
    c=customer_df["Cluster"]
)
ax1.set_xlabel("Total Quantity")
ax1.set_ylabel("Total Spending")
ax1.set_title("Customer Segmentation (K-Means)")
st.pyplot(fig1)

# =====================================================
# REGRESSION - RANDOM FOREST
# =====================================================
st.subheader("📈 Regression: Random Forest")

X = customer_df[["TotalQuantity"]]
y = customer_df["TotalSpending"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write("**R² Score:**", round(r2, 3))
st.write("**RMSE:**", round(rmse, 2))

# =====================================================
# VISUALISASI REGRESI
# =====================================================
fig2, ax2 = plt.subplots()
ax2.scatter(X_test, y_test, label="Actual", alpha=0.7)
ax2.scatter(X_test, y_pred, label="Predicted", alpha=0.7)
ax2.set_xlabel("Total Quantity")
ax2.set_ylabel("Total Spending")
ax2.set_title("Random Forest Regression")
ax2.legend()
st.pyplot(fig2)

# =====================================================
# USER INPUT - PREDICTION
# =====================================================
st.subheader("🔢 Input Data untuk Prediksi Total Spending")

with st.form("prediction_form"):
    qty_input = st.number_input(
        "Masukkan Total Quantity",
        min_value=1,
        value=10,
        step=1
    )
    submit = st.form_submit_button("Prediksi")

if submit:
    input_df = pd.DataFrame({
        "TotalQuantity": [qty_input]
    })

    prediction = rf.predict(input_df)

    st.success(f"💰 Prediksi Total Spending: {prediction[0]:,.2f}")

    # =================================================
    # VISUALISASI INPUT USER
    # =================================================
    fig3, ax3 = plt.subplots()
    ax3.scatter(
        customer_df["TotalQuantity"],
        customer_df["TotalSpending"],
        alpha=0.4,
        label="Data Asli"
    )
    ax3.scatter(
        qty_input,
        prediction,
        color="red",
        s=150,
        label="Input User"
    )
    ax3.set_xlabel("Total Quantity")
    ax3.set_ylabel("Total Spending")
    ax3.set_title("Visualisasi Prediksi Input User")
    ax3.legend()
    st.pyplot(fig3)

st.success("✅ Clustering, Regression, dan Input User berhasil dijalankan!")

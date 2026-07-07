import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="K-Means Clustering",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 K-Means Clustering using Iris Dataset")
st.write("This application demonstrates K-Means Clustering on the Iris dataset.")

# -----------------------------
# Load Dataset
# -----------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.subheader("📋 Iris Dataset")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
k = st.sidebar.slider("Select Number of Clusters (K)", 2, 8, 3)

# -----------------------------
# Feature Selection
# -----------------------------
X = df[["petal length (cm)", "petal width (cm)"]]

# -----------------------------
# Train K-Means Model
# -----------------------------
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X)

# -----------------------------
# Display Clustered Dataset
# -----------------------------
st.subheader("📊 Clustered Dataset")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Scatter Plot
# -----------------------------
fig = px.scatter(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    color=df["Cluster"].astype(str),
    title="K-Means Clustering Visualization",
    labels={
        "petal length (cm)": "Petal Length (cm)",
        "petal width (cm)": "Petal Width (cm)",
        "color": "Cluster"
    }
)

# Add Centroids
fig.add_trace(
    go.Scatter(
        x=model.cluster_centers_[:, 0],
        y=model.cluster_centers_[:, 1],
        mode="markers",
        marker=dict(
            size=18,
            color="black",
            symbol="star"
        ),
        name="Centroids"
    )
)

fig.update_layout(
    title_x=0.3,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Elbow Method
# -----------------------------
st.subheader("📈 Elbow Method")

sse = []

for i in range(1, 10):
    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    km.fit(X)
    sse.append(km.inertia_)

elbow_df = pd.DataFrame({
    "Number of Clusters": range(1, 10),
    "SSE": sse
})

fig2 = px.line(
    elbow_df,
    x="Number of Clusters",
    y="SSE",
    markers=True,
    title="Elbow Method for Optimal K"
)

fig2.update_layout(
    title_x=0.3,
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Cluster Centers
# -----------------------------
st.subheader("🎯 Cluster Centers")

centers = pd.DataFrame(
    model.cluster_centers_,
    columns=["Petal Length (cm)", "Petal Width (cm)"]
)

st.dataframe(centers, use_container_width=True)

# -----------------------------
# Summary
# -----------------------------
st.success(f"K-Means successfully created **{k} clusters** on the Iris dataset.")

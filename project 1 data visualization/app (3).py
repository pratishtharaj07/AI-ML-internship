import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Google Play Store Analysis",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Google Play Store App Analysis")

st.write("""
This project analyzes the Google Play Store dataset to understand
the factors that influence app performance and ratings.
""")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("googleplaystore_v2 (1).csv")

# -----------------------------
# Dataset
# -----------------------------
st.header("📋 Dataset")

st.dataframe(df)

# -----------------------------
# Dataset Information
# -----------------------------
st.header("📑 Dataset Information")

st.write("**Number of Rows:**", df.shape[0])
st.write("**Number of Columns:**", df.shape[1])

st.subheader("Column Data Types")
st.dataframe(df.dtypes)

# -----------------------------
# Data Cleaning
# -----------------------------
st.header("🧹 Data Cleaning")

# Missing values before cleaning
st.subheader("Missing Values Before Cleaning")
st.dataframe(df.isnull().sum())

# Remove rows with missing ratings
df = df[df["Rating"].notnull()]

# Convert Price column
if "Price" in df.columns:
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)

# Convert Reviews column
if "Reviews" in df.columns:
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# Convert Installs column
if "Installs" in df.columns:
    df["Installs"] = (
        df["Installs"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

# Convert Size (optional)
if "Size" in df.columns:
    df["Size"] = (
        df["Size"]
        .astype(str)
        .str.replace("M", "", regex=False)
        .str.replace("k", "", regex=False)
    )

# Missing values after cleaning
st.subheader("Missing Values After Cleaning")
st.dataframe(df.isnull().sum())

st.success("✅ Data Cleaning Completed Successfully")

# -----------------------------
# Visualizations
# -----------------------------
st.header("📊 Visualizations")

chart = st.selectbox(
    "Choose a Visualization",
    (
        "Rating Distribution",
        "Price Distribution",
        "Content Rating",
        "Top Categories",
        "Rating vs Reviews"
    )
)

# Rating Distribution
if chart == "Rating Distribution":

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(df["Rating"], bins=20, kde=True, ax=ax)

    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# Price Distribution
elif chart == "Price Distribution":

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(df["Price"], bins=20, ax=ax)

    ax.set_xlabel("Price")

    st.pyplot(fig)

# Content Rating
elif chart == "Content Rating":

    fig, ax = plt.subplots(figsize=(8,5))

    df["Content Rating"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Number of Apps")

    st.pyplot(fig)

# Top Categories
elif chart == "Top Categories":

    fig, ax = plt.subplots(figsize=(10,5))

    df["Category"].value_counts().head(10).plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Number of Apps")

    st.pyplot(fig)

# Rating vs Reviews
elif chart == "Rating vs Reviews":

    fig, ax = plt.subplots(figsize=(8,5))

    sns.scatterplot(
        data=df,
        x="Reviews",
        y="Rating",
        ax=ax
    )

    st.pyplot(fig)

# -----------------------------
# Summary Statistics
# -----------------------------
st.header("📈 Summary Statistics")

st.dataframe(df.describe())

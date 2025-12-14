import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier


# ============================================================
#  🔷 TITRE PRINCIPAL
# ============================================================
st.title("🤖 Modélisation & Évaluation")


# ============================================================
#  🔷 IMPORTATION & PRÉPARATION DES DONNÉES
# ============================================================

data = pd.read_csv("Financial_inclusion_dataset.csv")
df = data.copy()


# --- Nettoyage ---
df = df.drop('uniqueid', axis=1)

# Encodage
label_encoder = LabelEncoder()
for col in df.select_dtypes(exclude='number').columns:
    df[col] = label_encoder.fit_transform(df[col])

st.success("Variables catégorielles encodées et colonne inutile supprimée !")


# ============================================================
#  🔷 ANALYSE DE LA CORRÉLATION
# ============================================================
st.header("🔍 Analyse de la Corrélation ")

correlation = df.corr()

st.subheader("🔥 Heatmap des Corrélations")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(correlation, annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)


st.subheader("🎯 Corrélation avec la variable cible : `bank_account`")
correlation_target = correlation["bank_account"].sort_values()

fig2, ax2 = plt.subplots(figsize=(3, 4))
sns.heatmap(correlation.loc[correlation_target.index, ["bank_account"]], annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)


# --- Fonction pour sélectionner les colonnes corrélées ---
def get_correlated_columns(corr_df, target='bank_account', threshold=0.1, absolute=True):
    s = corr_df[target]
    if absolute:
        s = s.abs()
    s = s.drop(labels=[target], errors='ignore')
    return s[s >= threshold].sort_values(ascending=False)

seuil = 0.05
cols_correl = get_correlated_columns(correlation, threshold=seuil)
liste_cols = list(cols_correl.index)

st.write("### 🧩 Variables sélectionnées :")
st.table(liste_cols)


# ============================================================
#  🔷 SÉPARATION DES DONNÉES
# ============================================================
st.header("🧪 Sélection & Préparation des Données")

features = liste_cols
target = "bank_account"

X = df[features]
y = df[target]

st.write("Variables explicatives :", features)
st.write("Variable cible :", target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.info("Découpage effectué : 80% train / 20% test avec stratification.")


# ============================================================
#  🔷 SCALING
# ============================================================
st.subheader("⚙️ Mise à l'Échelle (StandardScaler)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.success("Données mises à l'échelle avec succès !")


# ============================================================
#  🔷 MODÉLISATION (XGBoost)
# ============================================================
st.header("🤖 Entraînement du Modèle XGBoost")

model = XGBClassifier(
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)
st.success("Modèle entraîné avec succès ! 🎯")


# ============================================================
#  🔷 ÉVALUATION DU MODÈLE
# ============================================================
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

st.header("📊 Évaluation du Modèle")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_probs)

st.markdown(f"#### 🔹 Accuracy : **{acc:.4f}**")
st.markdown(f"#### 🔹 Precision : **{prec:.4f}**")
st.markdown(f"#### 🔹 Recall : **{rec:.4f}**")
st.markdown(f"#### 🔹 F1-score : **{f1:.4f}**")
st.markdown(f"#### 🔹 AUC-ROC : **{auc:.4f}**")


# --- Matrice de confusion ---
st.subheader("📘 Matrice de Confusion")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.pyplot(fig_cm)


# --- Rapport de classification ---
st.subheader("📄 Rapport de Classification")
st.text(classification_report(y_test, y_pred))

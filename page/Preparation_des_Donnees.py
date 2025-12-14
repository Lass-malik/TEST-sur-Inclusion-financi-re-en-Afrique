import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io

data = pd.read_csv("Financial_inclusion_dataset.csv")
df = data.copy()

st.markdown("# Nettoyage & Encodage des Données")

st.markdown("## 🔧 Traitement des valeurs manquantes")
st.write("Les données sont déjà propres et ne contiennent pas de valeurs manquantes ni de valeurs aberrantes . Aucune action de nettoyage n'est nécessaire à ce stade.")

st.markdown("## 🔠 Encodage des variables catégorielles")
st.write("Les variables catégorielles ont été encodées avec le LabelEncoder de Scikit-Learn ")

#importation des données
data = pd.read_csv("Financial_inclusion_dataset.csv")

#Afficher les premières lignes du jeu de données
data.head()

# Découpagde des données
col_numeriques = data.select_dtypes(include=['int64', 'float64'])
col_categoriques = data.select_dtypes(include=['object']).drop(columns=['uniqueid']) # Suppression de la colonne 'uniqueid' qui n'est pas catégorique pertinente

st.markdown("* **Liste des colonnes numériques :**")
st.write(col_numeriques.columns.tolist())

st.markdown("* **Liste des colonnes catégorielles :**")
st.write(col_categoriques.columns.tolist())

df = data.copy()

#Suppression de la colonne 'user_id' qui n'est pas utile pour la modélisation
df = df.drop('uniqueid', axis=1)
#encodage par label encoding
label_encoder = LabelEncoder()
for col in df.select_dtypes(exclude='number').columns:
    df[col] = label_encoder.fit_transform(df[col])  

#Affichage des premières lignes du dataframe après traitement
st.markdown("## Aperçu des données après préparation")
st.write("La colonne 'uniqueid' a été supprimée et les variables catégorielles ont été encodées.")
st.dataframe(df)

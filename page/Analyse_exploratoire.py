import streamlit as st
import pandas as pd
import seaborn as sns
import io
import matplotlib.pyplot as plt

df = pd.read_csv("Financial_inclusion_dataset.csv")

st.title("Analyse Exploratoire des Données")

st.markdown("## 📊 Statistiques générale")

st.markdown("* **Aperçu des données**")
st.dataframe(df)

## Afficher les informations du DataFrame
st.markdown("* **Info dataset** (`df.info()`)")

# 1. Créer un buffer pour capturer la sortie
buffer = io.StringIO()

# 2. Exécuter df.info() en écrivant dans le buffer
# verbose=True assure que tous les détails sont inclus
df.info(buf=buffer, verbose=True) 

# 3. Récupérer le contenu du buffer
df_info_output = buffer.getvalue()

# 4. Afficher le contenu capturé dans un bloc de code
st.code(df_info_output, language='text')


st.markdown("* **Valeurs manquantes**")
st.write(" On remarque que le dataset contient des variables de types `object` (catégorielles) et `int64` (numériques). aucune colonnes n'a de valeurs manquantes qui devront être traitées lors de la préparation des données.")

st.markdown("* **Répartition de bank_account**")
# 1. Calcul de la répartition en pourcentages
bank_account_counts = df['bank_account'].value_counts()

# 2. Calcul des pourcentages pour l'affichage textuel
bank_account_percentages = (bank_account_counts / bank_account_counts.sum()) * 100

# 3. Créer une figure Matplotlib
fig, ax = plt.subplots(figsize=(2, 2))

# 4. Dessiner le graphique en secteurs (Pie Chart)
# autopct='%1.1f%%' ajoute les pourcentages sur les tranches
ax.pie(
    bank_account_counts.values, 
    labels=bank_account_counts.index, 
    autopct='%1.1f%%', 
    startangle=90,
    textprops={'fontsize': 6}
)
ax.axis('equal') # Assure que le graphique est un cercle
ax.set_title("Répartition de la variable 'bank_account'")

# 5. Afficher le graphique dans Streamlit
st.pyplot(fig)

st.markdown("On remarque que la majorité des individus n'ont pas de compte bancaire.")

st.markdown("## Distribution des variables clés")

st.markdown("* **age_of_respondent (histogramme)**")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['age_of_respondent'], bins=30, kde=True, ax=ax)
ax.set_title("Distribution de l'âge des répondants")
st.pyplot(fig)

st.markdown("* **household_size (histogramme)**")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df['household_size'], bins=20, kde=False, ax=ax)   
ax.set_title("Distribution de la taille des ménages")
st.pyplot(fig)
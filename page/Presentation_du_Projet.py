import streamlit as st
import pandas as pd
import io
# Charger les données pour afficher des informations sur le dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")

st.title("🏠 Accueil / Présentation du Projet")
st.subheader("Bienvenue dans notre application dédiée à la prédiction l'nclusion financière en Afrique!")

st.markdown('## 🎯 Objectif du projet')

st.write("Dans ce projet nous allons travailler sur les données **d'Inclusion financière** en Afrique qui a été fourni dans le cadre du projet **Inclusion financière en Afrique** hébergé par la plateforme Zindi. Description de l'ensemble de données : L'ensemble de données contient des informations démographiques et les services financiers utilisés par environ 33 600 personnes en Afrique de l'Est. Le rôle du modèle ML est de prédire quels individus sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire.")

st.markdown('## 📁 Structure des données')
st.markdown('##### Description rapide des colonnes du dataset :')
st.write("L'ensemble de données comprend plusieurs caractéristiques démographiques telles que l'âge, le sexe, le niveau d'éducation, l'état matrimonial, l'emploi, etc. La variable cible est 'has_account', qui indique si une personne possède un compte bancaire ou non.")

st.markdown('#### Nombre d’observations et de variables')
st.write("L'ensemble de données contient **23 524  observations** et **13 colonnes**.")


st.markdown('#### Type de variables')
# Utilisation d'un expander pour les détails, rendant l'affichage initial plus léger
st.subheader("1. Variables Catégorielles (Type: `object`)")

st.markdown("""
La majorité de vos colonnes sont catégorielles, représentant des étiquettes ou des identifiants textuels. 
Elles nécessiteront un **encodage** pour être utilisées en modélisation.
""")

with st.expander("Voir les détails et le traitement suggéré"):
    st.markdown("""
    * **Variables Nominales :** (Exemples : `country`, `marital_status`). Elles n'ont pas d'ordre. Elles nécessitent un **One-Hot Encoding**.
    * **Variables Ordinales :** (Exemple : `education_level`). Elles possèdent une hiérarchie naturelle. Elles nécessitent un **Ordinal Encoding**.
    * **Variables Binaires :** (Exemples : `bank_account`, `gender_of_respondent`). Elles ont seulement deux valeurs. Un simple **Label Encoding (0/1)** est suffisant.
    * **Identifiant :** (Exemple : `uniqueid`). Cette colonne est un identifiant unique et **doit être ignorée ou supprimée** pour la modélisation.
    """)

st.subheader("2. Variables Numériques (Type: `int64`)")

st.markdown("""
Ces variables représentent des quantités mesurables et sont généralement prêtes à l'emploi après une éventuelle mise à l'échelle.
""")

with st.expander("Voir les détails et la préparation"):
    st.markdown("""
    * **Variables Discrètes :** (Exemples : `household_size`, `year`). Ces variables résultent d'un comptage et peuvent être utilisées directement.
    * **Variables Quasi-Continues :** (Exemple : `age_of_respondent`). L'âge est souvent traité comme une variable **continue** ou **discrète**. Elle nécessitera une **mise à l'échelle** (`StandardScaler` ou `MinMaxScaler`) pour éviter de biaiser le modèle.
    """)



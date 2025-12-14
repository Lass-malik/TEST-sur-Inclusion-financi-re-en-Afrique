import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Les pages de l'application Streamlit seront définies ici
page1_Presentation_du_Projet =st.Page(
    page="page/Presentation_du_Projet.py",
    title="Présentation du Projet",
    icon="🏠",
    default=True,
)

page2_Analyse_exploratoire=st.Page(
    page="page/Analyse_exploratoire.py",
    title="Analyse exploratoire (EDA)",
    icon="🔍",
)

page3_Preparation_des_donnees=st.Page(
    page="page/Preparation_des_donnees.py",
    title="Préparation des données",
    icon="🛠️",
)

page4_Modelisation=st.Page(
    page="page/Modelisation.py",
    title="Modélisation & Évaluation",
    icon="🤖",
)

page5_prediction=st.Page(
    page="page/prediction.py",
    title="Prédiction",
    icon="🔮",
)

#Naviguer entre les pages
pg = st.navigation(
    pages={
            "Infos" :[page1_Presentation_du_Projet],
            
            "Projet": [ page2_Analyse_exploratoire, page3_Preparation_des_donnees, page4_Modelisation, page5_prediction],
    }
)

#Bas de page
st.sidebar.text("Développé par LASSISSI Malik © 2025")

#Afficher la page sélectionnée
pg.run()
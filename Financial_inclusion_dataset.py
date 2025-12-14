#1. Installer les paquets nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler



#importation des données
data = pd.read_csv("Financial_inclusion_dataset.csv")

#Afficher les premières lignes du jeu de données
data.head()
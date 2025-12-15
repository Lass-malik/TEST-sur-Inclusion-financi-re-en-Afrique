import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Prédiction Inclusion Financière", layout="wide")

st.title("🔮 Prédiction de l'inclusion financière")
st.write("Remplissez les informations ci-dessous pour obtenir la prédiction.")

# ============================================================
# 🔷 IMPORTATION ET PRÉPARATION DES DONNÉES
# ============================================================

data = pd.read_csv("Financial_inclusion_dataset.csv")
df = data.copy()
df = df.drop('uniqueid', axis=1)

# Encodage
label_encoder = LabelEncoder()
for col in df.select_dtypes(exclude='number').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Corrélation et sélection de features
corr = df.corr()
def get_correlated_columns(corr_df, target='bank_account', threshold=0.05):
    s = corr_df[target].abs()
    s = s.drop(labels=[target], errors='ignore')
    return s[s >= threshold].sort_values(ascending=False)

features = list(get_correlated_columns(corr).index)
target = 'bank_account'

X = df[features]
y = df[target]

# Découpage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle
model = XGBClassifier(
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ============================================================
# 🔷 MAPPING FRANÇAIS -> ANGLAIS
# ============================================================

mapping = {
    'education_level': {
        'Enseignement secondaire': 'Secondary education',
        'Pas d’éducation formelle': 'No formal education',
        'Formation professionnelle/spécialisée': 'Vocational/Specialised training',
        'Enseignement primaire': 'Primary education',
        'Enseignement supérieur': 'Tertiary education',
        'Autre/Ne sait pas/Refus': 'Other/Dont know/RTA'
    },
    'cellphone_access': {'Oui': 'Yes', 'Non': 'No'},
    'country': {'Kenya': 'Kenya', 'Rwanda': 'Rwanda', 'Tanzanie': 'Tanzania', 'Ouganda': 'Uganda'},
    'gender_of_respondent': {'Femme': 'Female', 'Homme': 'Male'},
    'location_type': {'Rural': 'Rural', 'Urbain': 'Urban'},
    'relationship_with_head': {
        'Conjoint(e)': 'Spouse',
        'Chef du ménage': 'Head of Household',
        'Autre parent': 'Other relative',
        'Enfant': 'Child',
        'Parent': 'Parent',
        'Autres non-parents': 'Other non-relatives'
    },
    'job_type': {
        'Travailleur indépendant': 'Self employed',
        'Dépendant du gouvernement': 'Government Dependent',
        'Employé formel privé': 'Formally employed Private',
        'Employé informel': 'Informally employed',
        'Employé formel gouvernement': 'Formally employed Government',
        'Agriculture et pêche': 'Farming and Fishing',
        'Dépendant des remises': 'Remittance Dependent',
        'Autres revenus': 'Other Income',
        'Ne sait pas / Refus': 'Dont Know/Refuse to answer',
        'Sans revenu': 'No Income'
    }
}

# ============================================================
# 🔷 FORMULAIRE PRINCIPAL
# ============================================================

st.header("📝 Formulaire de Prédiction")

# Valeurs pour les selectbox
education_level = list(mapping['education_level'].keys())
cellphone_access = list(mapping['cellphone_access'].keys())
country = list(mapping['country'].keys())
gender_of_respondent = list(mapping['gender_of_respondent'].keys())
year = [2018, 2016, 2017]
location_type = list(mapping['location_type'].keys())
relationship_with_head = list(mapping['relationship_with_head'].keys())
job_type = list(mapping['job_type'].keys())

# Utilisation de colonnes pour compacité
col1, col2 = st.columns(2)

with col1:
    q1 = st.selectbox("Quel est votre niveau d’éducation ?", education_level)
    q2 = st.selectbox("Avez-vous accès à un téléphone portable ?", cellphone_access)
    q3 = st.selectbox("Dans quel pays vivez-vous ?", country)
    q4 = st.selectbox("Quel est votre genre ?", gender_of_respondent)

with col2:
    q5 = st.selectbox("Année des informations ?", year)
    q6 = st.selectbox("Vivez-vous en zone urbaine ou rurale ?", location_type)
    q7 = st.selectbox("Lien avec le chef du ménage ?", relationship_with_head)
    q8 = st.selectbox("Type d’emploi ?", job_type)

# Bouton de prédiction
if st.button("Faire la prédiction"):

    user_input = pd.DataFrame({
        'education_level': [q1],
        'cellphone_access': [q2],
        'country': [q3],
        'gender_of_respondent': [q4],
        'year': [q5],
        'location_type': [q6],
        'relationship_with_head': [q7],
        'job_type': [q8]
    })

    # Appliquer le mapping français -> anglais
    for col in mapping.keys():
        user_input[col] = user_input[col].map(mapping[col])

    # Encoder selon dataset original
    for col in user_input.columns:
        if col in features:
            le = LabelEncoder()
            le.fit(data[col])
            user_input[col] = le.transform(user_input[col])

    # Sélection des features et scaling
    user_input = user_input[features]
    user_input_scaled = scaler.transform(user_input)

    # Prédiction
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)[:, 1]

    # Affichage
    st.markdown("## 📊 Résultat de la prédiction")
    st.write("**Inclusion financière prédite (compte bancaire) :**", "Oui" if int(prediction[0])==1 else "Non")
    st.write("**Probabilité associée :** {:.2f}%".format(float(prediction_proba[0])*100))
    if prediction[0] == 1:
        st.success("Félicitations ! Selon nos prédictions, vous êtes susceptible de posséder un compte bancaire.")
    else:
        st.warning("Selon nos prédictions, vous n'êtes probablement pas titulaire d'un compte bancaire.")

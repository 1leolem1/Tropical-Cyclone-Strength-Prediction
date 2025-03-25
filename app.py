import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF
import folium
from streamlit_folium import folium_static

# Charger le modèle
MODEL_PATH = "model.joblib"  # À remplacer par le chemin réel
model =1 # joblib.load(MODEL_PATH)

def transform_data(data):
    # Transformer les données pour correspondre aux attentes du modèle
    return data  # À adapter selon les besoins réels

def predict(data):
    transformed_data = transform_data(data)
    predictions = model.predict(transformed_data)
    return predictions

def generate_pdf(predictions_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Résumé des Prédictions", ln=True, align='C')
    
    for i, row in predictions_df.iterrows():
        pdf.cell(200, 10, txt=f"{row.to_dict()}", ln=True)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

st.title("Application de Prédiction")

# Choix de l'entrée : Saisie manuelle ou Upload CSV
input_mode = st.radio("Choisissez le mode d'entrée :", ["Saisie manuelle", "Fichier CSV"])

data = None
required_fields = ['WMO_WIND_ADJUSTED_COMPLETED', 'NATURE', 'USA_SSHS', 'age_hours', 'LON', 'LAT', 'STORM_SPEED']

if input_mode == "Saisie manuelle":
    input_values = {}
    for field in required_fields:
        input_values[field] = st.text_input(f"{field}")
    
    if st.button("Lancer la Prédiction"):
        data = pd.DataFrame([input_values])

elif input_mode == "Fichier CSV":
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        missing_columns = [col for col in required_fields if col not in data.columns]
        if missing_columns:
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {missing_columns}")
            data = None
        else:
            st.write("Aperçu des données :")
            st.dataframe(data.head())

if data is not None:
    predictions = predict(data)
    predictions_df = pd.DataFrame(predictions, columns=["Prédictions"])
    st.write("Résultats :")
    st.dataframe(predictions_df)
    
    csv_output = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger en CSV", csv_output, "predictions.csv", "text/csv")
    
    pdf_output = generate_pdf(predictions_df)
    st.download_button("Télécharger le résumé en PDF", pdf_output, "predictions.pdf", "application/pdf")
    
    # Affichage de la carte si les coordonnées sont disponibles
    if 'LAT' in data.columns and 'LON' in data.columns:
        st.write("Carte de la tempête :")
        m = folium.Map(location=[data['LAT'].mean(), data['LON'].mean()], zoom_start=5)
        for lat, lon in zip(data['LAT'], data['LON']):
            folium.Marker([lat, lon], popup="Tempête").add_to(m)
        folium_static(m)
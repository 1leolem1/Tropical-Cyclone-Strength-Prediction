import streamlit as st
import pandas as pd
import xgboost as xgb
import folium
import joblib
import time
from streamlit_folium import folium_static
from io import BytesIO

## Fonctions array 

def load_model():
    MODEL_PATH = "model_xgboost.pkl"  
    model = joblib.load(MODEL_PATH)  
    return model

model = load_model()

def read_data(data):
    df = pd.read_csv(data)

    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['YEAR'] = df['ISO_TIME'].dt.year
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['start_month'] = df.groupby('SID')['ISO_TIME'].transform('min').dt.month
    df['WMO_WIND'] = pd.to_numeric(df['WMO_WIND'], errors='coerce')
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df['DIST2LAND'] = pd.to_numeric(df['DIST2LAND'], errors='coerce')
    df.replace(to_replace=" ", value=pd.NA, inplace=True)

    return df

# Streamlit Interface 
st.set_page_config(page_title="Tropical Cyclone Prediction", layout="wide",page_icon='🌪️')
st.title("🌪️ Tropical Cyclone Strength Prediction")
tabs = st.tabs(["Introduction", "Prediction", "Map"])
# Main title (always visible)  

# Onglet Introduction
with tabs[0]:
    
      

    # Welcome Section  
    st.subheader("🚀 Welcome to our Streamlit app!")  
    st.write("""  
    Ever wondered how powerful a tropical cyclone could get? This app helps predict its severity using advanced machine learning!  
    """)  

    # About the Model  
    st.header("🔬 How does it work?")  
    st.write("""  
    We use **XGBoost**, a cutting-edge machine learning model trained on the **IBTrACS dataset**, a global collection of tropical cyclone data.  
    By analyzing key features like **wind speed, age, basin, and other environmental conditions**, the model estimates a cyclone’s intensity—from a mild **Tropical Depression** to a devastating **Category 5** storm. 🌊🌪️  
    """)  

    # How to Use  
    st.header("🛠️ How to use this app?")  
    st.write("""  
    1️⃣ Enter the cyclone’s meteorological data in the input fields.  
    2️⃣ Click the **Predict** button.  
    3️⃣ Instantly get a severity prediction along with key insights.  

    This can help **meteorologists, researchers, and disaster response teams** improve early warnings and better understand cyclone risks. 🏝️⚠️  
    """)  

    # Why Use This App?  
    st.header("🌍 Why use this app?")  
    st.write("""  
    ✅ **Easy to use** – Just enter the data, and get predictions in seconds!  
    ✅ **Powered by AI** – Trained on real-world cyclone data for **reliable results**.  
    ✅ **For everyone** – Whether you're a scientist or just curious, this app makes cyclone predictions accessible to all!  
    """)  

    # Call to Action  
    st.subheader("⚡ Ready to predict cyclone strength? Enter your data and get started!") 
# Onglet Prédiction
with tabs[1]:
    st.subheader("📂 Upload your dataset")  
    st.write("""  
    To predict the severity of tropical cyclones, please upload a CSV file containing meteorological data.  
    The model will analyze key features and estimate the cyclone’s intensity. 🚀  
    """)
    
    file = st.file_uploader("📥 Drag and drop your CSV file below", type=["csv"])
    if file:
        data = read_data(file)
        data['Prediction'] = 0
        success_message = st.empty()
        success_message.success("✅ Data successfully loaded!")
        time.sleep(1)
        success_message.empty()
        st.write("""  
            Click the button below to run the model. The system will process the data and return the predicted cyclone severity. 🌊  
            """)
        if st.button('⚡ Run Prediction'):
            
            with st.spinner('🔄 Calculating prediction... Please wait'):
                time.sleep(3)  # Simuler le délai de prédiction (remplace par ton code réel de prédiction)
                
                
                # Vérification des features du modèle
                required_features = model.get_booster().feature_names
                missing_features = [feature for feature in required_features if feature not in data.columns]
                
                if missing_features:
                    st.error(f"⚠️ The following columns are missing in the dataset: {missing_features}")
                else:
                    # Prédiction
                    predictions = model.predict(data[required_features])
                    data["Prediction"] = predictions
                    success_message = st.empty()
                    success_message.success("✅ Prediction complete!")
                    time.sleep(2)
                    success_message.empty()
                    
                    # Affichage des résultats  
                    st.subheader("📊 Prediction Results")  
                    st.write("Here are the predicted cyclone severity levels for your dataset:")
                    st.dataframe(
                        data[['LON','LAT','start_date','age_hours','Prediction']],
                        column_config={
                            "LON": '📍 Longitude',
                            "LAT": '📍 Latitude',
                            'start_date' : "📅 Start Date",
                            'age_hours' : st.column_config.NumberColumn(
                                                                "⏳ Age",
                                                                help="The age of the storm",
                                                                step=1,
                                                                format="%d hour(s)",
                                                            ),
                            "Prediction": st.column_config.NumberColumn(
                                                                "🔮 Prediction",
                                                                help="Predicted cyclone severity (0-7)",
                                                                min_value=0,
                                                                max_value=7,
                                                                step=1,
                                                                format="%d ⭐",
                                                            ),
                        },
                        hide_index=True,
                    )
                    
                    # Export des données  
                    st.subheader("📤 Export Results")  
                    st.write("Download the prediction results as a CSV file for further analysis.") 

                    # Export
                    buffer = BytesIO()
                    data.to_csv(buffer, index=False)
                    st.download_button("📥 Download CSV", buffer.getvalue(), file_name="predictions.csv", mime="text/csv")

with tabs[2]:
    # Titre principal avec icône
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🌍 Cyclone Location Map</h1>", unsafe_allow_html=True)

    # Vérification que le DataFrame est bien chargé
    if "data" in locals() and not data.empty:
        st.subheader("📍 View the cyclone locations on the map")
        st.write("""
        Here is an interactive map displaying the locations of the tropical cyclones based on the meteorological data you provided.
        The intensity of the cyclone is reflected in the color and size of each marker.
        """)

        # Création de la carte centrée sur le premier point
        center_lat, center_lon = data["LAT"].mean(), data["LON"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

        # Ajout des marqueurs
        for _, row in data.iterrows():
            lat, lon = row["LAT"], row["LON"]
            start_date = row["start_date"]
            age_hours = row["age_hours"]
            intensity = row["Prediction"]

            # Couleur de l'icône en fonction de l'intensité
            color_intensity = int(255 * (intensity / 7))  # Échelle de 0 à 255
            color = f"#{color_intensity:02x}0000"  # Rouge plus foncé pour intensité élevée

            # Ajout des cercles représentant chaque cyclone
            folium.CircleMarker(
                location=[lat, lon],
                radius=5 + (intensity * 0.5),  # Taille légèrement plus grande si intensité élevée
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"<b>Date:</b> {start_date}<br><b>Age:</b> {age_hours} hours<br><b>Intensity:</b> {intensity}⭐"
            ).add_to(m)

        # Affichage de la carte
        st.write("🌍 Here's the map with cyclone locations:")
        folium_static(m)

    else:
        st.warning("⚠️ No data available to display on the map.")

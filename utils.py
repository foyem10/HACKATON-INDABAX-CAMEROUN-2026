import joblib
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_resource
def load_model_and_scaler(model_name):
    """Charge le modèle et le scaler pour un nom donné (ex: 'hsi')"""
    model = joblib.load(f"models/{model_name}_model.pkl")
    scaler = joblib.load(f"models/{model_name}_scaler.pkl")
    features = joblib.load(f"models/{model_name}_features.pkl")  # liste des features
    return model, scaler, features

def compute_is_dry_season(month):
    """Fonction utilitaire (à adapter selon la définition)"""
    return 1 if month in [11,12,1,2,3] else 0
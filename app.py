import streamlit as st
import numpy as np
import joblib
import pandas as pd
import xgboost as xgb




model = joblib.load(r'Model/xgb_tuned.joblib')


# Function to preprocess and make predictions
def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)

# Streamlit app
def main():
    st.set_page_config(page_title="Patient Survival Prediction App", page_icon="🏥", layout="wide")



# Set use_label_encoder to False
    if hasattr(model, 'base_margin'):
    # Set base_margin to a dummy value
        model.base_margin = [0] * len(model.classes_)

    # Streamlit UI elements
    st.title("Patient Survival Prediction App 🏥")
    st.sidebar.header("Input Features")

   # Read feature names from the file
    with open("feature_names.txt", "r") as file:
        all_feature_names = [line.strip() for line in file]

    # Define default values for user inputs
    default_values = {
        'age': 25,
        'bmi': 22.5,
        'elective_surgery': 0,
        'height': 170.0,
        'pre_icu_los_days': 1.0,
        'weight': 70.0,
        'apache_2_diagnosis': 100.0,
        'apache_3j_diagnosis': 200.0,
        'apache_post_operative': 0,
        'arf_apache': 0,
        'bun_apache': 20.0,
        'creatinine_apache': 1.0,
        'gcs_eyes_apache': 4.0,
        'gcs_motor_apache': 6.0,
        'gcs_unable_apache': 0,
        'gcs_verbal_apache': 5.0,
        'glucose_apache': 120.0,
        'heart_rate_apache': 80.0,
        'hematocrit_apache': 40.0,
        'intubated_apache': 0,
        'map_apache': 90.0,
        'resprate_apache': 18.0,
        'sodium_apache': 138.0,
        'temp_apache': 37.0,
        'ventilated_apache': 0,
        'wbc_apache': 10.0,
        'd1_diasbp_max': 80.0,
        'd1_diasbp_min': 60.0,
        'd1_diasbp_noninvasive_max': 80.0,
        'd1_diasbp_noninvasive_min': 60.0,
        'd1_heartrate_max': 100.0,
        'd1_heartrate_min': 60.0,
        'd1_mbp_max': 90.0,
        'd1_mbp_min': 70.0,
        'd1_mbp_noninvasive_max': 90.0,
        'd1_mbp_noninvasive_min': 70.0,
        'd1_resprate_max': 20.0,
        'd1_resprate_min': 12.0,
        'd1_spo2_max': 98.0,
        'd1_spo2_min': 95.0,
        'd1_sysbp_max': 120.0,
        'd1_sysbp_min': 80.0,
        'd1_sysbp_noninvasive_max': 120.0,
        'd1_sysbp_noninvasive_min': 80.0,
        'd1_temp_max': 37.5,
        'd1_temp_min': 36.5,
        'h1_diasbp_max': 80.0,
        'h1_diasbp_min': 60.0,
        'h1_diasbp_noninvasive_max': 80.0,
        'h1_diasbp_noninvasive_min': 60.0,
        'h1_heartrate_max': 100.0,
        'h1_heartrate_min': 60.0,
        'h1_mbp_max': 90.0,
        'h1_mbp_min': 70.0,
        'h1_mbp_noninvasive_max': 90.0,
        'h1_mbp_noninvasive_min': 70.0,
        'h1_resprate_max': 20.0,
        'h1_resprate_min': 12.0,
        'h1_spo2_max': 98.0,
        'h1_spo2_min': 95.0,
        'h1_sysbp_max': 120.0,
        'h1_sysbp_min': 80.0,
        'h1_sysbp_noninvasive_max': 120.0,
        'h1_sysbp_noninvasive_min': 80.0,
        'h1_temp_max': 37.5,
        'h1_temp_min': 36.5,
        'd1_bun_max': 25.0,
        'd1_bun_min': 10.0,
        'd1_calcium_max': 10.5,
        'd1_calcium_min': 8.5,
        'd1_creatinine_max': 1.0,
        'd1_creatinine_min': 0.5,
        'd1_glucose_max': 120.0,
        'd1_glucose_min': 70.0,
        'd1_hco3_max': 30.0,
        'd1_hco3_min': 22.0,
        'd1_hemaglobin_max': 17.0,
        'd1_hemaglobin_min': 12.0,
        'd1_hematocrit_max': 50.0,
        'd1_hematocrit_min': 38.0,
        'd1_platelets_max': 400.0,
        'd1_platelets_min': 150.0,
        'd1_potassium_max': 5.0,
        'd1_potassium_min': 3.5,
        'd1_sodium_max': 145.0,
        'd1_sodium_min': 135.0,
        'd1_wbc_max': 12.0,
        'd1_wbc_min': 4.0,
        'apache_4a_hospital_death_prob': 0.05,
        'apache_4a_icu_death_prob': 0.03,
        'aids': 0,
        'cirrhosis': 0,
        'diabetes_mellitus': 0,
        'hepatic_failure': 0,
        'immunosuppression': 0,
        'leukemia': 0,
        'lymphoma': 0,
        'solid_tumor_with_metastasis': 0,
        'isin_african': 0,
        'isin_asian': 0,
        'isin_caucasian': 0,
        'isin_hispanic': 0,
        'isin_native': 0,
        'isin_other/unknown': 0,
        'isin_f': 0,
        'isin_m': 0,
        'isin_ccu-cticu': 0,
        'isin_csicu': 0,
        'isin_cticu': 0,
        'isin_cardiac': 0,
        'isin_micu': 0,
        'isin_med-surg': 0,
        'isin_neuro': 0,
        'isin_sicu': 0,
        'isin_cardiovascular': 0,
        'isin_gastrointestinal': 0,
        'isin_genitourinary': 0,
        'isin_gynecological': 0,
        'isin_hematological': 0,
        'isin_neurological': 0,
        'isin_respiratory': 0,
        'isin_sepsis': 0,
        'isin_trauma': 0,
        'isin_haematologic': 0,
        'isin_neurologic': 0,
        'isin_renal/genitourinary': 0,
        'isin_undefined': 0,
    
    
    }

    # Define user inputs for each feature
    user_inputs = {}
    for feature in all_feature_names:
        user_inputs[feature] = st.sidebar.text_input(feature, default_values.get(feature, ""))

    submit_button = st.sidebar.button("Predict")

    if submit_button:
        # Prepare the input data for prediction
        data = np.array([user_inputs[feature] for feature in all_feature_names]).reshape(1, -1)


        # Make predictions
        pred = get_prediction(data=data, model=model)
        st.write(f"The predicted survival probability i:  {pred[0]}")
        

if __name__ == "__main__":
    main()

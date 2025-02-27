import pandas as pd
import streamlit as st
from transformers import pipeline

# Dictionary to explain new risk factors
RISK_FACTORS = {
    "Smokes": "Smoking increases the risk of cervical cancer by making it harder for the body to fight HPV infections.",
    "Hormonal Contraceptives": "Long-term use of hormonal contraceptives can slightly increase the risk of cervical cancer.",
    "IUD": "Intrauterine devices (IUDs) may influence cervical cancer risk, though some studies suggest they might offer some protection.",
    "STDs": "Sexually transmitted diseases (STDs), especially HPV, are a major risk factor for cervical cancer.",
    "Dx:Cancer": "A previous cancer diagnosis may affect overall health and risk factors.",
    "Dx:HPV": "A positive HPV diagnosis means there is a higher risk of developing cervical cancer over time."
}

# Load transformer model for text generation
generator = pipeline('text-generation', model='gpt2')  # Correct model identifier

def generate_risk_explanation(row):
    """Generate a risk explanation based on user inputs."""
    explanation = []
    
    # Age factor
    explanation.append(f"🔍 Age Factor: Being {row['Age']} years old, your risk level depends on other factors.")
    
    # Sexual activity risks
    if row['Number of sexual partners'] > 4:
        explanation.append("• Having multiple sexual partners can increase the risk of HPV exposure.")
    if row['STDs:HPV'] == 'Yes':
        explanation.append("• Since you have tested positive for HPV, regular screenings are important.")
    
    # Smoking risk
    if row['Smokes'] == 'Yes':
        explanation.append(f"• {RISK_FACTORS['Smokes']}")
    
    # Hormonal contraceptives
    if row['Hormonal Contraceptives'] == 'Yes' and row['Hormonal Contraceptives (years)'] > 5:
        explanation.append(f"• {RISK_FACTORS['Hormonal Contraceptives']}")
    
    # IUD risk
    if row['IUD'] == 'Yes':
        explanation.append(f"• {RISK_FACTORS['IUD']}")
    
    # STDs risk
    if row['STDs'] == 'Yes' or row['STDs:HPV'] == 'Yes':
        explanation.append(f"• {RISK_FACTORS['STDs']}")
    
    return "\n".join(explanation)

def generate_detailed_explanation(feature, importance):
    """Generate a detailed explanation for a feature using the transformer model."""
    prompt = f"The feature '{feature}' has an importance score of {importance:.4f} in predicting cervical cancer risk. Explain why this feature is important and how it affects the risk."
    detailed_explanation = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return detailed_explanation

def main():
    st.title("Cervical Cancer Risk & Report Explainer")
    st.write("Understand your risk factors and medical report in simple terms. 🏥")
    
    # Patient Inputs
    age = st.number_input("Age:", min_value=10, max_value=100, value=30)
    num_partners = st.number_input("Number of sexual partners:", min_value=0, max_value=50, value=1)
    first_sex = st.number_input("Age at first sexual intercourse:", min_value=10, max_value=50, value=18)
    num_pregnancies = st.number_input("Number of pregnancies:", min_value=0, max_value=20, value=0)
    
    # Lifestyle & Medical History
    smokes = st.selectbox("Do you smoke?", ["Yes", "No"])
    smokes_years = st.number_input("Years of smoking:", min_value=0, max_value=50, value=0) if smokes == "Yes" else 0
    hormonal_contraceptives = st.selectbox("Have you used hormonal contraceptives?", ["Yes", "No"])
    hc_years = st.number_input("Years using hormonal contraceptives:", min_value=0, max_value=50, value=0) if hormonal_contraceptives == "Yes" else 0
    iud = st.selectbox("Have you used an IUD?", ["Yes", "No"])
    iud_years = st.number_input("Years using an IUD:", min_value=0, max_value=50, value=0) if iud == "Yes" else 0
    
    # STDs & Diagnosis
    stds = st.selectbox("Have you had any STDs?", ["Yes", "No"])
    stds_hpv = st.selectbox("Have you tested positive for HPV?", ["Yes", "No"])
    dx_cancer = st.selectbox("Previous cancer diagnosis?", ["Yes", "No"])
    dx_hpv = st.selectbox("Previous HPV diagnosis?", ["Yes", "No"])
    
    # Create DataFrame
    data = pd.DataFrame({
        'Age': [age],
        'Number of sexual partners': [num_partners],
        'First sexual intercourse': [first_sex],
        'Num of pregnancies': [num_pregnancies],
        'Smokes': [smokes],
        'Smokes (years)': [smokes_years],
        'Hormonal Contraceptives': [hormonal_contraceptives],
        'Hormonal Contraceptives (years)': [hc_years],
        'IUD': [iud],
        'IUD (years)': [iud_years],
        'STDs': [stds],
        'STDs:HPV': [stds_hpv],
        'Dx:Cancer': [dx_cancer],
        'Dx:HPV': [dx_hpv]
    })
    
    if st.button("Explain My Report"):
        st.subheader("📋 Your Personalized Risk Explanation")
        explanation = generate_risk_explanation(data.iloc[0])
        st.markdown(explanation)
        
        # Display detailed feature explanations
        st.subheader("📊 Detailed Feature Explanations")
        features = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:HPV']
        importances = [0.150, 0.125, 0.100, 0.075, 0.200, 0.175, 0.050, 0.100, 0.150]  # Example importances
        
        for feature, importance in zip(features, importances):
            detailed_explanation = generate_detailed_explanation(feature, importance)
            st.markdown(f"**{feature}**: {detailed_explanation}")
        
if __name__ == "__main__":
    main()
# Required imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import shap
import warnings
warnings.filterwarnings('ignore')

# Helper Functions
def format_medical_term(term):
    """Format medical terms for better readability"""
    term = term.replace('_', ' ').title()
    term = term.replace('Std', 'STD')
    term = term.replace('Iud', 'IUD')
    term = term.replace('Hpv', 'HPV')
    return term
# Add this method to the CervicalCancerAnalyzer class
def generate_transformer_explanation(self, feature_importance_dict):
    """Generate detailed explanations using the Transformer model."""
    try:
        from transformers import pipeline
        
        # Load the GPT-2 model
        generator = pipeline('text-generation', model='gpt2')
        
        explanations = []
        for feature, importance in feature_importance_dict.items():
            prompt = f"The feature '{feature}' has an importance score of {importance:.4f} in predicting cervical cancer risk. Explain why this feature is important and how it affects the risk."
            detailed_explanation = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            explanations.append(f"**{feature}**: {detailed_explanation}")
        
        return "\n\n".join(explanations)
    
    except Exception as e:
        return f"Error generating Transformer explanation: {str(e)}"
def get_medical_explanation(factor):
    """Provide detailed medical explanation for different risk factors"""
    explanations = {
        'age': {
            'mechanism': "Age affects cervical cancer risk due to accumulated exposure to HPV (human papillomavirus) and cellular changes over time. The cervix undergoes transformation zones at different life stages, making certain age groups more susceptible to abnormal cell development.",
            'risk_patterns': "Risk typically increases starting from age 30-35, with peak incidence between 45-60. Younger women tend to clear HPV infections more effectively due to stronger immune responses.",
            'medical_context': "Regular screening becomes especially important as age increases, with different screening intervals recommended for different age groups based on risk profiles."
        },
        'smoking': {
            'mechanism': "Smoking introduces carcinogenic chemicals that can damage the DNA of cervical cells. Tobacco by-products have been found in the cervical mucus of smokers, directly exposing cervical tissue to harmful compounds.",
            'risk_patterns': "Both active and passive smoking increase risk, with a dose-dependent relationship between number of cigarettes and cancer risk. Smoking also suppresses the immune system's ability to clear HPV infections.",
            'medical_context': "Smoking cessation can significantly reduce risk over time, though some damage may persist. Combined with other risk factors, smoking can multiplicatively increase cancer risk."
        },
        'sexual': {
            'mechanism': "Multiple sexual partners increase exposure to HPV, the primary cause of cervical cancer. Early sexual activity can affect still-developing cervical cells, making them more susceptible to abnormal changes.",
            'risk_patterns': "Risk increases with number of partners and earlier age of first intercourse. This relates to increased HPV exposure and potential tissue damage during cervical development.",
            'medical_context': "Safe sex practices and regular screening are especially important for those with multiple partners or early sexual debut."
        },
        'hormonal': {
            'mechanism': "Hormonal contraceptives can influence the cervical epithelium and hormone-dependent gene expression. Long-term use may affect how cervical cells respond to damage and potentially influence cancer development.",
            'risk_patterns': "Risk may increase with long-term hormonal contraceptive use (>5 years), though modern formulations have lower risks than older ones.",
            'medical_context': "Benefits of hormonal contraception often outweigh risks, but regular screening is essential for long-term users."
        },
        'STD': {
            'mechanism': "STDs can cause inflammation and tissue damage, making cervical cells more susceptible to abnormal changes. Some STDs can directly damage cellular DNA or weaken local immune responses.",
            'risk_patterns': "Multiple or recurrent STDs increase risk significantly, especially when combined with HPV infection. Chronic inflammation from untreated STDs is particularly concerning.",
            'medical_context': "Regular STD screening and prompt treatment are crucial for risk reduction. Some STDs can have long-term effects even after treatment."
        },
        'pregnancy': {
            'mechanism': "Pregnancy-related hormonal changes and cervical trauma can affect cell development. Multiple pregnancies may increase exposure to hormonal fluctuations and physical stress on cervical tissue.",
            'risk_patterns': "High parity (multiple full-term pregnancies) may increase risk, possibly due to hormonal factors and cervical trauma during delivery.",
            'medical_context': "Regular prenatal care and pap smears during pregnancy are important for early detection. Post-pregnancy follow-up is also crucial."
        },
        'IUD': {
            'mechanism': "IUDs may affect the local cervical environment and influence cellular changes. Different types of IUDs (hormonal vs. copper) may have different effects on cervical tissue.",
            'risk_patterns': "Modern IUDs show minimal risk increase, though long-term studies are ongoing. Local inflammation may play a role in any observed risk changes.",
            'medical_context': "Regular check-ups and screening are important for IUD users, though modern IUDs are generally considered safe."
        }
    }
    return explanations.get(factor, {
        'mechanism': "This factor's biological mechanism requires further research.",
        'risk_patterns': "Risk patterns are still being studied.",
        'medical_context': "Medical implications are under investigation."
    })

def answer_question(question, model, feature_names):
    """Answer questions about how factors affect cervical cancer risk with detailed medical explanations"""
    try:
        # Get feature importances from the model
        importances = model.feature_importances_
        feature_importance = dict(zip(feature_names, importances))
        
        # List of keywords to match in questions
        keywords = {
            'age': ['age', 'old', 'young'],
            'smoking': ['smoke', 'smoking', 'cigarette'],
            'sexual': ['sexual', 'intercourse', 'partners'],
            'hormonal': ['hormonal', 'contraceptive', 'birth control'],
            'STD': ['std', 'sexually transmitted', 'infection'],
            'pregnancy': ['pregnancy', 'pregnant', 'pregnancies'],
            'cancer': ['cancer', 'tumor', 'malignant'],
            'IUD': ['iud', 'intrauterine']
        }
        
        # Find relevant factors based on question
        relevant_factors = []
        for key, words in keywords.items():
            if any(word.lower() in question.lower() for word in words):
                relevant_factors.extend([f for f in feature_names if key.lower() in f.lower()])
        
        if not relevant_factors:
            return "I'm not sure about that specific factor. Please ask about age, smoking, sexual history, hormonal contraceptives, pregnancies, or STDs."
        
        # Generate detailed response
        response = "Based on our analysis and medical research:\n\n"
        found_significant = False
        
        for factor in relevant_factors:
            importance = feature_importance[factor]
            if importance > 0.01:  # Only include factors with >1% importance
                found_significant = True
                
                # Determine which category this factor belongs to
                factor_category = next((k for k, v in keywords.items() if any(w in factor.lower() for w in v)), None)
                if factor_category:
                    medical_info = get_medical_explanation(factor_category)
                    
                    response += f"## {format_medical_term(factor)}\n\n"
                    response += f"Statistical Importance: This factor accounts for {importance:.1%} of the model's decision-making.\n\n"
                    response += f"Biological Mechanism: {medical_info['mechanism']}\n\n"
                    response += f"Risk Patterns: {medical_info['risk_patterns']}\n\n"
                    response += f"Medical Context: {medical_info['medical_context']}\n\n"
        
        if not found_significant and relevant_factors:
            response += "While these factors are tracked in our data, they don't show a strong statistical relationship with cervical cancer risk in this dataset.\n\n"
        
        response += "Important Note: These findings are based on statistical analysis and medical research, but individual cases may vary. "
        response += "Always consult with healthcare professionals for personalized medical advice and screening recommendations."
        
        return response
        
    except Exception as e:
        return f"Error answering question: {str(e)}"

class CervicalCancerExplainer:
    """Class to generate explanations for model predictions"""
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Initialize LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            class_names=['No Cancer', 'Cancer'],
            mode='classification'
        )
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(model)
    
    def explain_prediction(self, patient_data):
        """Explain prediction for a single patient"""
        try:
            # Get LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                patient_data.values[0], 
                self.model.predict_proba,
                num_features=10
            )
            
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(patient_data)
            
            # Create natural language explanation
            prediction = self.model.predict(patient_data)[0]
            proba = self.model.predict_proba(patient_data)[0]
            
            explanation = self._generate_natural_language_explanation(
                lime_exp,
                shap_values,
                prediction,
                proba,
                patient_data
            )
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def _generate_natural_language_explanation(self, lime_exp, shap_values, 
                                             prediction, proba, patient_data):
        """Generate natural language explanation of the prediction"""
        risk_level = "high" if prediction == 1 else "low"
        confidence = proba[1] if prediction == 1 else proba[0]
        
        # Get top factors from LIME
        features = dict(lime_exp.as_list())
        
        explanation = f"\nBased on the analysis of your medical history, you appear to be at {risk_level} "
        explanation += f"risk for cervical cancer (confidence: {confidence:.2%}).\n\n"
        explanation += "The most important factors in this assessment are:\n\n"
        
        for feature, impact in features.items():
            if abs(impact) > 0.01:  # Only include significant factors
                direction = "increasing" if impact > 0 else "decreasing"
                explanation += f"- Your {format_medical_term(feature)} is {direction} your risk\n"
        
        explanation += "\nPlease note that this is a statistical analysis and should not "
        explanation += "replace professional medical advice. Consult with your healthcare "
        explanation += "provider for proper medical diagnosis and treatment options."
        
        return explanation

class CervicalCancerAnalyzer:
    """Main class for cervical cancer risk analysis"""
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.imputer = None
        self.explainer = None

    def load_data(self, file_path):
        """Load and validate the dataset"""
        try:
            self.df = pd.read_csv(file_path)
            print("Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            self._validate_data()
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def _validate_data(self):
        """Validate data quality and print summary"""
        print("\nData Summary:")
        print("-" * 50)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nData Types:")
        print(self.df.dtypes)
        
        print("\nValue Counts for Target (Biopsy):")
        print(self.df['Biopsy'].value_counts(normalize=True))

    def preprocess_data(self):
        """Preprocess the data for modeling"""
        try:
            # Replace '?' with NaN
            self.df = self.df.replace('?', np.nan)
            
            # Convert all columns to numeric
            for col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Separate features and target
            self.X = self.df.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1)
            self.y = self.df['Biopsy']
            
            # Store feature names
            self.feature_names = self.X.columns.tolist()
            
            # Handle missing values
            self.imputer = SimpleImputer(strategy='median')
            self.X = pd.DataFrame(
                self.imputer.fit_transform(self.X),
                columns=self.X.columns
            )
            
            # Scale the features
            self.scaler = StandardScaler()
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns
            )
            
            print("Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return False

    def train_model(self):
        """Train the Random Forest model"""
        try:
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(self.X_train, self.y_train)
            
            # Initialize explainer
            self.explainer = CervicalCancerExplainer(
                self.model,
                self.X_train,
                self.feature_names
            )
            
            # Evaluate model
            self._evaluate_model()
            
            return True
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return False

    def _evaluate_model(self):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("-" * 50)
        
        # Calculate scores
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"Training Score: {train_score:.4f}")
        print(f"Testing Score: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        print(f"\n5-Fold Cross-validation Scores: {cv_scores}")
        print(f"Average CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        y_pred = self.model.predict(self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            print("Error: Model not trained yet!")
            return
        
        plt.figure(figsize=(12, 8))
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importances.head(15))
        plt.title('Top 15 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(15, 12))
        correlation_matrix = self.X.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self):
        """Plot ROC curve"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def answer_question(self, question):
        """Answer questions about risk factors"""
        if self.model is None:
            print("Error: Model not trained yet!")
            return None
            
        return answer_question(question, self.model, self.feature_names)

    def analyze_age_groups(self):
        """Analyze risk patterns across age groups"""
        age_groups = pd.qcut(self.df['Age'], q=5)
        age_risk = pd.DataFrame({
            'Age_Group': age_groups,
            'Biopsy': self.df['Biopsy']
        })
        
        risk_by_age = age_risk.groupby('Age_Group')['Biopsy'].mean()
        
        plt.figure(figsize=(10, 6))
        risk_by_age.plot(kind='bar')
        plt.title('Risk of Cervical Cancer by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Risk Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return risk_by_age

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CervicalCancerAnalyzer()
    
    # Load and prepare data
    file_path = "path_to_your_data.csv"
    if analyzer.load_data(file_path):
        if analyzer.preprocess_data():
            if analyzer.train_model():
                # Example: Ask a question
                question = "How does age affect cervical cancer risk?"
                print(analyzer.answer_question(question))
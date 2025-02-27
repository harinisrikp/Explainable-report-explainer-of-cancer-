from cervical_cancer_analyzer import CervicalCancerAnalyzer

def main():
    # Initialize analyzer
    analyzer = CervicalCancerAnalyzer()

    # Load and analyze data
    analyzer.load_data("D:/mini project/CC 36/kag_risk_factors_cervical_cancer.csv")
    analyzer.preprocess_data()
    analyzer.train_model()

    # Get feature importances from the RF model
    feature_importance_dict = dict(zip(analyzer.feature_names, analyzer.model.feature_importances_))

    # Generate explanations using the Transformer model
    transformer_explanation = analyzer.generate_transformer_explanation(feature_importance_dict)
    print("Transformer Model Explanations:")
    print(transformer_explanation)

    # Ask questions
    question = "How does age affect cervical cancer risk?"
    print(analyzer.answer_question(question))

if __name__ == "__main__":
    main()
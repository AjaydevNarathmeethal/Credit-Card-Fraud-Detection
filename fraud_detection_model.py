import joblib
import gradio as gr
import pandas as pd

model = joblib.load("best_model.pkl")

feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                   'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                   'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                   'V28', 'Amount']

def predict_model(features_str):
    try:
        data_values = [float(x.strip()) for x in features_str.split(',')]
        data = pd.DataFrame([data_values], columns=feature_columns)

    except ValueError:
        return "Invalid input! Please enter numbers separated by commas.", 0.0

    if data.shape[1] != len(feature_columns):
       return f"Expected {len(feature_columns)} features, got {len(data)}.", 0.0

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1].round(5)
    # Color-coded output
    # Format and color-code the output
    if prediction == 1:
        output_html = f"""
        <div style='text-align:center; color:white; background-color:#ff4d4d; 
                    font-weight:bold; font-size:1.3em; padding:15px; 
                    border-radius:10px;'>
            ❌ Fraud Detected! <br>
            Fraud Probability: {prob:.2%}
        </div>
        """
    else:
        output_html = f"""
        <div style='text-align:center; color:white; background-color:#28a745; 
                    font-weight:bold; font-size:1.3em; padding:15px; 
                    border-radius:10px;'>
            ✅ Not Fraud <br>
            Fraud Probability: {prob:.2%}
        </div>
        """

    return output_html

inputs = gr.Textbox(label=f"Enter {len(feature_columns)} features separated by commas", 
                    placeholder="12, 2, 3, 4, ...")

iface = gr.Interface(
    fn=predict_model,
    inputs=inputs,
    outputs=gr.HTML(label="Result"),
    title="Fraud Detection Model",
    description="Enter transaction features to get prediction from the best model."
)

iface.launch()
import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("../model/loan_model.pkl")

# Prediction function
def predict_loan(gender, married, dependents, education, self_employed,
                 applicant_income, coapplicant_income, loan_amount,
                 loan_term, credit_history, property_area):

    # Encode categorical inputs manually as per LabelEncoder
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0

    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents = dependents_map.get(dependents, 0)

    property_area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_area = property_area_map.get(property_area, 0)

    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area]])

    prediction = model.predict(input_data)[0]
    return "✅ Approved" if prediction == 1 else "❌ Not Approved"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 Loan Approval Predictor")
    gr.Markdown("Enter the applicant details below to check loan eligibility:")

    with gr.Row():
        gender = gr.Radio(["Male", "Female"], label="Gender")
        married = gr.Radio(["Yes", "No"], label="Married")
        dependents = gr.Dropdown(["0", "1", "2", "3+"], label="Dependents")
        education = gr.Radio(["Graduate", "Not Graduate"], label="Education")

    with gr.Row():
        self_employed = gr.Radio(["Yes", "No"], label="Self Employed")
        credit_history = gr.Radio(["Yes", "No"], label="Credit History")
        property_area = gr.Dropdown(["Rural", "Semiurban", "Urban"], label="Property Area")

    with gr.Row():
        applicant_income = gr.Number(label="Applicant Income")
        coapplicant_income = gr.Number(label="Coapplicant Income")

    with gr.Row():
        loan_amount = gr.Number(label="Loan Amount")
        loan_term = gr.Number(label="Loan Term (in days)")

    submit_btn = gr.Button("Predict Loan Status")
    result = gr.Textbox(label="Prediction")

    submit_btn.click(fn=predict_loan,
                     inputs=[gender, married, dependents, education, self_employed,
                             applicant_income, coapplicant_income, loan_amount,
                             loan_term, credit_history, property_area],
                     outputs=result)

# Run the app
demo.launch()

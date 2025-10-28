
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.preprocessing import LabelEncoder

# Try to locate the model and training CSV reliably (support a couple of relative paths)
ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [ROOT / "model" / "loan_model.pkl", ROOT.parent / "model" / "loan_model.pkl"]
DATA_PATHS = [ROOT / "data" / "train.csv", ROOT.parent / "data" / "train.csv"]

model_path = next((p for p in MODEL_PATHS if p.exists()), None)
data_path = next((p for p in DATA_PATHS if p.exists()), None)

if model_path is None:
    raise FileNotFoundError(
        "Could not find model file. Expected at one of: "
        + ", ".join(str(p) for p in MODEL_PATHS)
    )
if data_path is None:
    raise FileNotFoundError(
        "Could not find training data (train.csv). Expected at one of: "
        + ", ".join(str(p) for p in DATA_PATHS)
    )

# Load model
model = joblib.load(model_path)

# Load training data to rebuild encoders and imputing values
train_df = pd.read_csv(data_path)

# Reproduce preprocessing from notebook:
# - drop Loan_ID
# - fill missing values (modes for categorical; median for LoanAmount; mode for Loan_Amount_Term, Credit_History)
train_df = train_df.copy()
if "Loan_ID" in train_df.columns:
    train_df.drop("Loan_ID", axis=1, inplace=True)

# Imputation statistics
_impute = {}
# Categorical columns that were label encoded in notebook
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

for col in ["Gender", "Married", "Dependents", "Self_Employed"]:
    if col in train_df.columns:
        _impute[col] = train_df[col].mode()[0]
for col in ["Loan_Amount_Term", "Credit_History"]:
    if col in train_df.columns:
        _impute[col] = train_df[col].mode()[0]
if "LoanAmount" in train_df.columns:
    _impute["LoanAmount"] = train_df["LoanAmount"].median()

# Apply imputations to training copy (so encoders see the same unique values)
for k, v in _impute.items():
    if k in train_df.columns:
        train_df[k].fillna(v, inplace=True)

# Build LabelEncoders exactly as in the notebook (fit on training data after imputation)
encoders = {}
le = LabelEncoder()
for col in categorical_cols:
    if col in train_df.columns:
        enc = LabelEncoder()
        # Fill any remaining NA just in case
        train_df[col] = train_df[col].fillna(_impute.get(col, ""))
        enc.fit(train_df[col].astype(str))
        encoders[col] = enc

# Feature order used in notebook/model: Gender, Married, Dependents, Education, Self_Employed,
# ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
FEATURE_ORDER = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


def preprocess_input(input_dict):
    """
    input_dict: mapping of raw user inputs (strings/numbers).
    Returns a 2D array (1 x n_features) ready for model.predict.
    """
    # Start with DataFrame to make transforms simpler
    df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)

    # Impute missing numeric/categorical if left blank
    for col, val in _impute.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Ensure numeric types for numeric columns
    numeric_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in numeric_cols:
        if col in df.columns:
            # Convert empty strings to NaN then fill
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in _impute:
                df[col].fillna(_impute[col], inplace=True)
            else:
                # fallback
                df[col].fillna(0, inplace=True)

    # Encode categorical columns using the encoders built from train.csv
    for col in categorical_cols:
        if col in df.columns:
            val = df.at[0, col]
            # If user input is numeric-like for categorical, convert to string consistently
            if pd.isna(val):
                val = _impute.get(col, "")
            val_str = str(val)
            enc = encoders.get(col)
            if enc is None:
                # fallback: simple label encoding by factorizing
                df[col] = pd.factorize(df[col].astype(str))[0]
            else:
                # If the value wasn't seen in training, we need to handle it.
                if val_str not in enc.classes_:
                    # Append unseen label temporarily to transform
                    # sklearn LabelEncoder doesn't support unseen labels, so we map to a special code.
                    # We'll map unknown -> len(classes_) which is a new integer not seen during training.
                    # But model wasn't trained with that code; better approach is to map unknown -> most frequent class code (mode)
                    most_freq = enc.transform([train_df[col].mode()[0]])[0] if col in train_df.columns else 0
                    df[col] = most_freq
                else:
                    df[col] = enc.transform([val_str])[0]

    # Order columns
    df = df[FEATURE_ORDER]
    return df.values


def predict_loan(
    Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area,
):
    """
    Inputs come from the UI. Return prediction label and probability (if available).
    """
    input_dict = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }

    X = preprocess_input(input_dict)
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        return f"Error during prediction: {e}"

    # If model was trained with LabelEncoder for Loan_Status, 1 -> approved, 0 -> not approved
    # Notebook used LabelEncoder for Loan_Status; we assume same mapping. We'll interpret numeric 1 as "Approved".
    label = "Approved" if int(pred) == 1 else "Not Approved"

    prob_text = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # If binary, proba[1] is probability for class 1 (approved)
        if proba.shape[0] >= 2:
            approved_prob = proba[1]
            prob_text = f" (probability Approved: {approved_prob:.2f})"
    return f"{label}{prob_text}"


# Build Gradio UI
gender_choices = sorted(train_df["Gender"].astype(str).unique().tolist()) if "Gender" in train_df.columns else ["Male", "Female"]
married_choices = sorted(train_df["Married"].astype(str).unique().tolist()) if "Married" in train_df.columns else ["Yes", "No"]
dependents_choices = sorted(train_df["Dependents"].astype(str).unique().tolist()) if "Dependents" in train_df.columns else ["0", "1", "2", "3+"]
education_choices = sorted(train_df["Education"].astype(str).unique().tolist()) if "Education" in train_df.columns else ["Graduate", "Not Graduate"]
self_emp_choices = sorted(train_df["Self_Employed"].astype(str).unique().tolist()) if "Self_Employed" in train_df.columns else ["Yes", "No"]
property_choices = sorted(train_df["Property_Area"].astype(str).unique().tolist()) if "Property_Area" in train_df.columns else ["Urban", "Rural", "Semiurban"]

with gr.Blocks() as demo:
    gr.Markdown("# Loan Approval Predictor")
    gr.Markdown("Enter applicant information and get a loan approval prediction.")

    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(label="Gender", choices=gender_choices, value=gender_choices[0])
            married = gr.Dropdown(label="Married", choices=married_choices, value=married_choices[0])
            dependents = gr.Dropdown(label="Dependents", choices=dependents_choices, value=dependents_choices[0])
            education = gr.Dropdown(label="Education", choices=education_choices, value=education_choices[0])
            self_emp = gr.Dropdown(label="Self Employed", choices=self_emp_choices, value=self_emp_choices[0])
            property_area = gr.Dropdown(label="Property Area", choices=property_choices, value=property_choices[0])

        with gr.Column():
            applicant_income = gr.Number(label="Applicant Income", value=3000)
            coapplicant_income = gr.Number(label="Coapplicant Income", value=0)
            loan_amount = gr.Number(label="Loan Amount (in thousands)", value=100)
            loan_term = gr.Number(label="Loan Amount Term (in days)", value=360)
            credit_history = gr.Number(label="Credit History (1.0 = good, 0.0 = bad)", value=1.0)

    predict_btn = gr.Button("Predict")
    output = gr.Textbox(label="Prediction")

    predict_btn.click(
        predict_loan,
        inputs=[
            gender,
            married,
            dependents,
            education,
            self_emp,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area,
        ],
        outputs=output,
    )

# Provide a few examples (optional)
example_inputs = [
    ["Male", "Yes", "0", "Graduate", "No", 5000, 0, 128, 360, 1.0, "Urban"],
    ["Female", "No", "1", "Not Graduate", "No", 3000, 1500, 66, 360, 1.0, "Rural"],
]
demo.launch(server_name="0.0.0.0", server_port=7860)

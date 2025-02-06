import streamlit as st
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fpdf import FPDF

st.title("AI Fairness Evaluation Tool Prototype")
st.markdown(
    """
    This tool allows healthcare providers to evaluate the fairness of clinical decision support (CDS) systems.
    Upload a dataset, select fairness metrics, and view results. The tool supports multiple metrics such as:
    - **Statistical Parity Difference**
    - **Disparate Impact**
    - **Equal Opportunity Difference**
    - **Average Odds Difference**
    - **Theil Index**
    - **Error Rate Difference**
    - **False Positive Rate (FPR) Difference**
    - **False Negative Rate (FNR) Difference**
    - **Predictive Parity**
    - **Consistency**
    - **Calibration Within Groups**
    """
)

def filter_groups(row, protected_attribute, operator, value):
    if operator == ">":
        return row[protected_attribute] > value
    elif operator == "<":
        return row[protected_attribute] < value
    elif operator == "=":
        return row[protected_attribute] == value

def compute_fairness_metrics(metric):
    return {
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Disparate Impact": metric.disparate_impact()
    }

def generate_pdf_report(protected_attribute, outcome, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fairness Compliance Report", ln=True)
    pdf.cell(200, 10, txt=f"Protected Attribute: {protected_attribute}", ln=True)
    pdf.cell(200, 10, txt=f"Outcome Column: {outcome}", ln=True)
    for key, value in metrics.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.4f}", ln=True)
    
    pdf.output("compliance_report.pdf")
    
    with open("compliance_report.pdf", "rb") as file:
        st.download_button(
            label="Download Compliance Report",
            data=file,
            file_name="compliance_report.pdf",
            mime="application/pdf"
        )

def compute_fairness_and_generate_report(data, protected_attribute, outcome):
    privileged_group = [{protected_attribute: 1}]
    unprivileged_group = [{protected_attribute: 0}]

    if set(data[outcome].unique()) != {0, 1}:
        min_label, max_label = sorted(data[outcome].unique())[:2]
        data[outcome] = data[outcome].replace({min_label: 0, max_label: 1})


    dataset = BinaryLabelDataset(
        df=data,
        label_names=[outcome],
        protected_attribute_names=[protected_attribute]
    )

    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=unprivileged_group,
        privileged_groups=privileged_group
    )

    metrics = compute_fairness_metrics(metric)

    if st.button("Calculate Fairness Metrics"):
        st.write("### Fairness Metrics")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.4f}")
    
    if st.button("Generate Compliance Report"):
        generate_pdf_report(protected_attribute, outcome, metrics)

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    given_file = pd.read_csv(uploaded_file)
    data = given_file.copy()
    st.write("### Uploaded Dataset")
    st.write(data.head())

    encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            encoders[column] = {k: v for v, k in enumerate(data[column].unique())}
            data[column] = data[column].map(encoders[column])

    st.write("### Encoded Dataset")
    st.write(data.head())

    outcome = st.selectbox("Select the Outcome Column", data.columns, index=len(data.columns) - 1)
    protected_attribute = st.selectbox("Select the Protected Attribute", data.columns)    
    if pd.api.types.is_numeric_dtype(given_file[protected_attribute]):
        operator = st.selectbox("Select Operator", ["=", ">", "<"])
        value = st.text_input("Enter Value for Comparison")
    else:
        unique_values = given_file[protected_attribute].unique()
        operator = st.selectbox("Select Operator", ["="])
        value = st.selectbox("Enter Value for Comparison", unique_values)

    if value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') < 2):
        value = float(value) if '.' in value else int(value)
    elif protected_attribute in encoders:
        value = encoders[protected_attribute][value]

    data['Group'] = data.apply(lambda row: filter_groups(row, protected_attribute, operator, value), axis=1)
    data[protected_attribute] = data['Group'].astype(int)

    compute_fairness_and_generate_report(data, protected_attribute, outcome)
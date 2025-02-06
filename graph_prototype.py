import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fpdf import FPDF

# Title and Description
st.title("Fairness Evaluation Tool Prototype")
st.markdown(
    """
    This tool allows healthcare providers to evaluate the fairness of clinical decision support (CDS) systems.
    Upload a dataset, select fairness metrics, and view results.
    """
)

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.write(data.head())

    # Encode non-numeric categorical data
    encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':
            encoders[column] = {k: v for v, k in enumerate(data[column].unique())}
            data[column] = data[column].map(encoders[column])

    st.write("### Encoded Dataset")
    st.write(data.head())

    # Step 2: Select Protected Attribute and Outcome
    protected_attribute = st.selectbox("Select the Protected Attribute", data.columns)
    outcome = st.selectbox("Select the Outcome Column", data.columns, index=len(data.columns) - 1)

    # Fairness Metric Selection
    metric_options = {
        "Statistical Parity Difference": "statistical_parity_difference",
        "Disparate Impact": "disparate_impact",
        "Equal Opportunity Difference": "equal_opportunity_difference",
        "Average Odds Difference": "average_odds_difference"
    }
    selected_metrics = st.multiselect("Select Fairness Metrics", list(metric_options.keys()), default=["Statistical Parity Difference"])

    # Step 3: Fairness Evaluation
    dataset = BinaryLabelDataset(
        df=data,
        label_names=[outcome],
        protected_attribute_names=[protected_attribute]
    )

    if st.button("Calculate Fairness Metrics"):
        metrics_by_group = {}

        unique_values = data[protected_attribute].unique()
        for group in unique_values:
            privileged_group = [{protected_attribute: group}]
            unprivileged_group = [{protected_attribute: val} for val in unique_values if val != group]

            metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_group, unprivileged_groups=unprivileged_group)
            
            group_metrics = {metric_name: getattr(metric, metric_func)() for metric_name, metric_func in metric_options.items() if metric_name in selected_metrics}
            metrics_by_group[group] = group_metrics

        # Display results
        st.write("### Fairness Metrics by Group")
        metric_df = pd.DataFrame(metrics_by_group).T
        st.write(metric_df)

        st.write("### Fairness Metrics Visualization")
        fig, ax = plt.subplots(figsize=(max(10, len(unique_values) * 0.4), 6))  # Dynamically adjust figure size
        metric_df.plot(kind='bar', ax=ax)
        
        ax.set_xlabel("Groups")
        ax.set_ylabel("Metric Value")
        ax.set_title("Fairness Metric Comparisons")
        
        ax.set_xticks(range(len(unique_values)))
        ax.set_xticklabels([str(val) for val in unique_values], rotation=45, ha="right")

        plt.tight_layout()
        st.pyplot(fig)

    # if st.button("Generate Compliance Report"):
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", size=12)
    #     pdf.cell(200, 10, txt="Fairness Compliance Report", ln=True)
    #     pdf.cell(200, 10, txt=f"Protected Attribute: {protected_attribute}", ln=True)
    #     pdf.cell(200, 10, txt=f"Outcome Column: {outcome}", ln=True)

    #     for group, metrics in metrics_by_group.items():
    #         decoded_group = next((key for key, value in encoders[protected_attribute].items() if value == group), group)
    #         pdf.cell(200, 10, txt=f"Group: {protected_attribute} = {decoded_group}", ln=True)
    #         for metric_name, value in metrics.items():
    #             pdf.cell(200, 10, txt=f"{metric_name}: {value:.4f}", ln=True)

    #     pdf.output("compliance_report.pdf")

    #     with open("compliance_report.pdf", "rb") as file:
    #         st.download_button(
    #             label="Download Compliance Report",
    #             data=file,
    #             file_name="compliance_report.pdf",
    #             mime="application/pdf"
    #         )

    st.write("### Decoded Dataset for Reference")
    decoded_data = data.copy()
    for column, mapping in encoders.items():
        decoded_data[column] = decoded_data[column].map({v: k for k, v in mapping.items()})
    st.write(decoded_data.head())
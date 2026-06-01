import os

import streamlit as st

from src.Pipeline.predict_pipeline import CustomData, PredictPipeline


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.11), transparent 28rem),
            linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
        color: #172033;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #172033;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #667085;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #0f766e 100%);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4vw, 3.4rem);
        margin: 0 0 0.65rem;
    }

    .hero p {
        color: #dbeafe !important;
        font-size: 1.05rem;
        max-width: 760px;
        margin: 0;
    }

    .stSelectbox label,
    .stNumberInput label {
        color: #344054 !important;
        font-weight: 650;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: white;
        border: 1px solid #d8e0ea;
        border-radius: 8px;
        color: #172033;
    }

    div.stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        min-height: 3rem;
        width: 100%;
        font-size: 1rem;
        font-weight: 750;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
    }

    div.stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d8e0ea;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }

    hr {
        border-color: #d8e0ea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Customer Churn")
    st.caption("Telecom retention model")

    st.markdown("---")
    st.subheader("Model Performance")
    st.write("Model: Logistic Regression")
    st.write("Accuracy: 73.95%")
    st.write("F1 Score: 61.49%")
    st.write("ROC AUC: 84.11%")

    st.markdown("---")
    st.info("Enter a customer's profile to estimate their churn risk.")


st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>
            Estimate churn probability from customer profile, account services,
            billing preferences, and contract details.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    with col2:
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict Churn")


if submitted:
    try:
        data = CustomData(
            gender=gender,
            SeniorCitizen=SeniorCitizen,
            Partner=Partner,
            Dependents=Dependents,
            tenure=tenure,
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges,
        )

        pred_df = data.get_data_as_dataframe()
        prediction, probability = PredictPipeline().predict(pred_df)

        churn_probability = float(probability[0]) * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        result_col, detail_col = st.columns([1, 1.3], gap="large")

        with result_col:
            st.metric("Churn Probability", f"{churn_probability:.2f}%")
            st.progress(min(int(churn_probability), 100))

            if churn_probability >= 70:
                st.error(f"High churn risk ({churn_probability:.2f}%)")
            elif churn_probability >= 40:
                st.warning(f"Medium churn risk ({churn_probability:.2f}%)")
            else:
                st.success(f"Low churn risk ({churn_probability:.2f}%)")

        with detail_col:
            st.subheader("Customer Summary")
            st.json(
                {
                    "Gender": gender,
                    "Tenure": tenure,
                    "Contract": Contract,
                    "Monthly Charges": MonthlyCharges,
                    "Total Charges": TotalCharges,
                    "Internet Service": InternetService,
                }
            )

        if os.path.exists("artifacts/roc_curve.png"):
            st.markdown("---")
            st.subheader("ROC Curve")
            st.image("artifacts/roc_curve.png", use_container_width=True)

    except Exception as e:
        st.exception(e)


if os.path.exists("artifacts/metrics.txt"):
    st.markdown("---")
    st.subheader("Model Metrics")

    with open("artifacts/metrics.txt", "r", encoding="utf-8") as file:
        st.text(file.read())


st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit.")
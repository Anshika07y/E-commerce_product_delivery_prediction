import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="E-Commerce Delivery Prediction",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

model = joblib.load("decision_tree_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")

# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================

st.sidebar.title("🤖 Model Information")

st.sidebar.markdown("""
### Final Model
**Decision Tree Classifier**

### Model Type
Supervised Machine Learning  
Binary Classification

### Target Variable
`Reached.on.Time_Y.N`

- **1 = Reached On Time**
- **0 = Not Reached On Time**
""")

st.sidebar.divider()

st.sidebar.subheader("📊 Model Performance")

st.sidebar.metric(
    "Accuracy",
    "65.09%"
)

st.sidebar.metric(
    "Precision",
    "70.08%"
)

st.sidebar.metric(
    "Recall",
    "72.43%"
)

st.sidebar.metric(
    "F1-Score",
    "71.24%"
)

st.sidebar.divider()

st.sidebar.info(
    "The model was trained using the E-Commerce Product "
    "Delivery dataset and the same preprocessing pipeline "
    "used during model development."
)

# ============================================================
# MAIN TITLE
# ============================================================

st.title("📦 E-Commerce Product Delivery Prediction")

st.markdown(
    """
    ### Predict whether an e-commerce product will reach the customer on time

    Enter the product, customer, and shipment details below.
    The trained Decision Tree model will generate a prediction
    along with the estimated probability for each delivery outcome.
    """
)

st.divider()

# ============================================================
# PRODUCT DETAILS
# ============================================================

st.header("📋 Product & Shipment Details")

col1, col2 = st.columns(2)

with col1:

    warehouse_block = st.selectbox(
        "Warehouse Block",
        ["A", "B", "C", "D", "F"]
    )

    mode_of_shipment = st.selectbox(
        "Mode of Shipment",
        ["Ship", "Flight", "Road"]
    )

    product_importance = st.selectbox(
        "Product Importance",
        ["low", "medium", "high"]
    )

    cost_of_product = st.number_input(
        "Cost of the Product",
        min_value=96,
        max_value=310,
        value=210,
        step=1
    )

    discount_offered = st.number_input(
        "Discount Offered",
        min_value=1,
        max_value=65,
        value=13,
        step=1
    )

with col2:

    gender = st.selectbox(
        "Gender",
        ["F", "M"]
    )

    customer_care_calls = st.number_input(
        "Customer Care Calls",
        min_value=2,
        max_value=7,
        value=4,
        step=1
    )

    customer_rating = st.number_input(
        "Customer Rating",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    prior_purchases = st.number_input(
        "Prior Purchases",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

    weight_in_gms = st.number_input(
        "Weight in Grams",
        min_value=1001,
        max_value=7846,
        value=3600,
        step=1
    )

st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Delivery Status",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Warehouse_block": [warehouse_block],
        "Mode_of_Shipment": [mode_of_shipment],
        "Product_importance": [product_importance],
        "Gender": [gender],
        "Customer_care_calls": [customer_care_calls],
        "Customer_rating": [customer_rating],
        "Cost_of_the_Product": [cost_of_product],
        "Prior_purchases": [prior_purchases],
        "Discount_offered": [discount_offered],
        "Weight_in_gms": [weight_in_gms]
    })

    # --------------------------------------------------------
    # APPLY SAME PREPROCESSING AS TRAINING
    # --------------------------------------------------------

    input_encoded = preprocessor.transform(input_data)

    input_scaled = scaler.transform(input_encoded)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]

    # Get probability for both classes
    probabilities = model.predict_proba(input_scaled)[0]

    not_on_time_probability = probabilities[0] * 100
    on_time_probability = probabilities[1] * 100

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.divider()

    st.header("🎯 Prediction Result")

    if prediction == 1:

        st.success(
            "✅ The product is predicted to reach on time."
        )

    else:

        st.warning(
            "⚠️ The product is predicted not to reach on time."
        )

    # --------------------------------------------------------
    # PROBABILITY METRICS
    # --------------------------------------------------------

    st.subheader("📊 Prediction Probability")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:

        st.metric(
            "✅ Probability of Reaching On Time",
            f"{on_time_probability:.2f}%"
        )

        st.progress(
            int(round(on_time_probability))
        )

    with prob_col2:

        st.metric(
            "⚠️ Probability of Not Reaching On Time",
            f"{not_on_time_probability:.2f}%"
        )

        st.progress(
            int(round(not_on_time_probability))
        )

    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    st.divider()

    st.subheader("📋 Entered Product Details")

    display_data = pd.DataFrame({
        "Feature": [
            "Warehouse Block",
            "Mode of Shipment",
            "Product Importance",
            "Gender",
            "Customer Care Calls",
            "Customer Rating",
            "Cost of Product",
            "Prior Purchases",
            "Discount Offered",
            "Weight in Grams"
        ],
        "Value": [
            warehouse_block,
            mode_of_shipment,
            product_importance,
            gender,
            customer_care_calls,
            customer_rating,
            cost_of_product,
            prior_purchases,
            discount_offered,
            weight_in_gms
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.subheader("ℹ️ Prediction Interpretation")

    if prediction == 1:

        st.info(
            f"The model predicts **On Time** with an estimated "
            f"probability of **{on_time_probability:.2f}%**."
        )

    else:

        st.info(
            f"The model predicts **Not On Time** with an estimated "
            f"probability of **{not_on_time_probability:.2f}%**."
        )

    st.caption(
        "Prediction probabilities are generated by the trained "
        "Decision Tree model and should be interpreted as model "
        "confidence estimates rather than guarantees."
    )
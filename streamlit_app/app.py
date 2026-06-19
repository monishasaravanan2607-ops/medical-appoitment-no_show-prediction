import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

from datetime import timedelta

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Medical Appointment No-Show Prediction & Demand Forecasting Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ==================================================
# LOAD MODELS
# ==================================================

df = pd.read_csv("/Users/kaviraj/Desktop/GUVI/Project3/regression/r_feature_engineering.csv")
df["appointment_date_continuous"] = pd.to_datetime(df["appointment_date_continuous"])

xgb_model = joblib.load(
    "/Users/kaviraj/Desktop/GUVI/Project3/streamlit_app/xbg_classifier.pkl"
)

lr_model = joblib.load(
    "/Users/kaviraj/Desktop/GUVI/Project3/streamlit_app/lr_demand_forecast_model.pkl"
)

classification_cols = joblib.load(
    "/Users/kaviraj/Desktop/GUVI/Project3/streamlit_app/xgb_classification_columns.pkl"
)

regression_cols = joblib.load(
    "/Users/kaviraj/Desktop/GUVI/Project3/streamlit_app/lr_columns.pkl"
)



# ==================================================
# HEADER
# ==================================================

st.title("🏥 Medical Appointment No-Show Prediction & Demand Forecasting Dashboard")

st.markdown(
    """
    Predict patient attendance and forecast future appointment demand.
    """
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/hospital-3.png",
    width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "No-Show Classification",
        "Demand Forecasting"
    ]
)

# ==================================================
# NO SHOW CLASSIFICATION
# ==================================================

if page == "No-Show Classification":

    st.header("📅 Patient No-Show Prediction")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        "XGBoost"
    )

    c2.metric(
        "Purpose",
        "Attendance Prediction"
    )

    c3.metric(
        "Department",
        "Outpatient"
    )

    st.divider()

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Patient Age",
            0,
            100,
            30
        )

        gender = st.selectbox(
            "Gender",
            ["F", "M", "I"]
        )

        appointment_time = st.slider(
            "Appointment Hour",
            0,
            23,
            10
        )

        specialty = st.selectbox(
            "Specialty",
            [
                "assist",
                "enf",
                "occupational therapy",
                "pedagogo",
                "physiotherapy",
                "psychotherapy",
                "sem especialidade",
                "speech therapy"
            ]
        )

    with col2:
        companion = st.selectbox(
        "Needs Companion",
        [0, 1]
        )

        alcoholism = st.selectbox(
        "Alcoholism",
        [0, 1]
        )

        handicap = st.selectbox(
        "Handicap",
        [0, 1]
        )
        sms_received = st.selectbox(
            "SMS Reminder Sent",
            [0, 1]
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1]
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0, 1]
        )

        scholarship = st.selectbox(
            "Scholarship",
            [0, 1]
        )

    with st.expander(
        "🌦 Weather Information"
    ):

        avg_temp = st.number_input(
            "Average Temperature",
            value=25.0
        )

        avg_rain = st.number_input(
            "Average Rainfall",
            value=0.0
        )

        max_temp = st.number_input(
            "Maximum Temperature",
            value=30.0
        )

        max_rain = st.number_input(
            "Maximum Rainfall",
            value=0.0
        )

        rainy_day_before = st.selectbox(
            "Rainy Day Before",
            [0, 1]
        )

        storm_day_before = st.selectbox(
            "Storm Day Before",
            [0, 1]
        )
    
    # Hidden Features

    under_12 = 1 if age < 12 else 0
    over_60 = 1 if age > 60 else 0

    disability_intellectual = 0
    disability_motor = 0

    rain_intensity = avg_rain
    heat_intensity = avg_temp

    morning_shift = (
        1 if appointment_time < 12 else 0
    )
    day_of_week = st.selectbox(
    "Day Of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)
    if st.button(
    "🔍 Predict Attendance",
    use_container_width=True
    ):

        data = pd.DataFrame(
            0,
            index=[0],
            columns=classification_cols
        )

        data["appointment_time"] = appointment_time
        data["age"] = age

        data["under_12_years_old"] = under_12
        data["over_60_years_old"] = over_60

        data["patient_needs_companion"] = companion

        data["average_temp_day"] = avg_temp
        data["average_rain_day"] = avg_rain

        data["max_temp_day"] = max_temp
        data["max_rain_day"] = max_rain

        data["rainy_day_before"] = rainy_day_before
        data["storm_day_before"] = storm_day_before

        data["rain_intensity"] = rain_intensity
        data["heat_intensity"] = heat_intensity

        data["Hipertension"] = hypertension
        data["Diabetes"] = diabetes
        data["Alcoholism"] = alcoholism
        data["Handcap"] = handicap

        data["Scholarship"] = scholarship

        data["SMS_received"] = sms_received

        data["disability_intellectual"] = disability_intellectual
        data["disability_motor"] = disability_motor

        data["appointment_shift_morning"] = morning_shift

        if gender == "M":
            data["gender_M"] = 1

        elif gender == "I":
            data["gender_I"] = 1

        specialty_col = f"specialty_{specialty}"

        if specialty_col in data.columns:
            data[specialty_col] = 1
        
        day_col = f"day_of_week_{day_of_week}"

        if day_col in data.columns:
            data[day_col] = 1

        prediction = xgb_model.predict(data)
        prob = xgb_model.predict_proba(data)

        st.divider()

        st.subheader("🎯 Prediction Result")

        if prediction[0] == 1:
            st.warning(
                f"⚠️ Patient likely to miss appointment\n\nRisk Score: {prob[0][1]*100:.2f}%"
            )
        else:
            st.success(
                f"✅ Patient likely to attend appointment\n\nAttendance Score: {prob[0][0]*100:.2f}%"
            )
        # ==================================================
        # RISK PROBABILITY CHART
        # ==================================================
            
        st.subheader("📊 Risk Probability")

        risk_df = pd.DataFrame({
            "Outcome": ["Show", "No Show"],
            "Probability": [prob[0][0], prob[0][1]]
        })

        st.bar_chart(
            risk_df.set_index("Outcome")
        )

        # ==================================================
        # PREDICTION DETAILS
        # ==================================================

        c1, c2 = st.columns(2)

        c1.metric(
            "Show Probability",
            f"{prob[0][0]*100:.2f}%"
        )

        c2.metric(
            "No-Show Probability",
            f"{prob[0][1]*100:.2f}%"
        )

        # FEATURE IMPORTANCE
        st.subheader("📊 Top 10 Important Features")

        importance = pd.DataFrame({
            "Feature": classification_cols,
            "Importance": xgb_model.feature_importances_
        })
        importance = (
            importance
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Important Features - XGBoost")

        ax.invert_yaxis()

        st.pyplot(fig)

        # ==================================================
        # BUSINESS INSIGHTS
        # ==================================================

        with st.expander("📌 Business Insights"):
            st.markdown("""
                **Key Observations**
                
                - Patients with previous risk factors are more likely to miss appointments.
                - Weather conditions may influence attendance.
                - SMS reminders can improve attendance rates.
                - Age and medical conditions contribute to no-show behavior.
                - Appointment timing can affect attendance probability.
            """)

# ==================================================
# DEMAND FORECASTING
# ==================================================

else:

    st.header("📈 Appointment Demand Forecasting")

    # Load forecast dataset
    df = pd.read_csv("/Users/kaviraj/Desktop/GUVI/Project3/regression/r_feature_engineering.csv")

    df["appointment_date_continuous"] = pd.to_datetime(
        df["appointment_date_continuous"]
    )

    # -----------------------------------
    # Create March Test Set
    # -----------------------------------

    test_df = df[
        (df["appointment_date_continuous"] >= "2021-03-01") &
        (df["appointment_date_continuous"] < "2021-04-01")
    ]

    X_test = test_df[regression_cols]

    y_test = test_df["daily_appointments"]

    y_pred_lr = lr_model.predict(X_test)

    comparison_df = pd.DataFrame({
        "Date": test_df["appointment_date_continuous"],
        "Actual": y_test.values,
        "Predicted": np.round(y_pred_lr)
    })

    # -----------------------------------
    # Specialty Filter
    # -----------------------------------

    specialities = {
        "psychotherapy": "specialty_psychotherapy",
        "speech therapy": "specialty_speech therapy",
        "physiotherapy": "specialty_physiotherapy",
        "occupational therapy": "specialty_occupational therapy",
        "Unknown": "specialty_Unknown"
    }

    selected_speciality = st.selectbox(
        "Select Specialty",
        list(specialities.keys())
    )

    selected_date = st.date_input(
        "Select March 2021 Date",
        value=pd.to_datetime("2021-03-01"),
        min_value=pd.to_datetime("2021-03-01"),
        max_value=pd.to_datetime("2021-03-31")
    )

    forecast_days = st.selectbox(
        "Forecast Horizon",
        [3, 7, 15]
    )

    # ==================================================
    # FORECAST FUNCTION
    # ==================================================

    def forecast_demand(
        selected_speciality,
        forecast_days,
        selected_date
    ):

        speciality_col = specialities[selected_speciality]

        speciality_df = df[
            df[speciality_col] == 1
        ]

        current_date = pd.Timestamp(selected_date)

        historical_df = speciality_df[
            speciality_df[
                "appointment_date_continuous"
            ] <= current_date
        ]

        if historical_df.empty:
            return pd.DataFrame()

        latest_row = historical_df.sort_values(
            "appointment_date_continuous"
        ).iloc[-1]

        lag1 = latest_row["lag_1"]
        lag7 = latest_row["lag_7"]
        lag30 = latest_row["lag_30"]

        rolling7 = latest_row["rolling_mean_7"]
        rolling30 = latest_row["rolling_mean_30"]

        history = [lag30] * 23 + [lag7] * 6 + [lag1]

        predictions = []

        for i in range(1, forecast_days + 1):

            future_date = current_date + timedelta(days=i)

            row = {}

            row["lag_1"] = lag1
            row["lag_7"] = lag7
            row["lag_30"] = lag30

            row["rolling_mean_7"] = rolling7
            row["rolling_mean_30"] = rolling30

            # Specialty columns

            for col in specialities.values():
                row[col] = 0

            row[speciality_col] = 1

            # Weekday columns

            row["day_of_week_Monday"] = int(
                future_date.day_name() == "Monday"
            )

            row["day_of_week_Saturday"] = int(
                future_date.day_name() == "Saturday"
            )

            row["day_of_week_Sunday"] = int(
                future_date.day_name() == "Sunday"
            )

            row["day_of_week_Thursday"] = int(
                future_date.day_name() == "Thursday"
            )

            row["day_of_week_Tuesday"] = int(
                future_date.day_name() == "Tuesday"
            )

            row["day_of_week_Wednesday"] = int(
                future_date.day_name() == "Wednesday"
            )

            X_future = pd.DataFrame([row])

            X_future = X_future.reindex(
                columns=regression_cols,
                fill_value=0
            )

            prediction = lr_model.predict(
                X_future
            )[0]

            predictions.append({
                "Date": future_date.date(),
                "Predicted Appointments": max(
                    0,
                    round(prediction)
                ),
                "Availability":
                "Available"
                if round(prediction) > 0
                else "No Availability"
            })

            history.append(prediction)

            lag1 = prediction
            lag7 = history[-7]
            lag30 = history[-30]

            rolling7 = np.mean(
                history[-7:]
            )

            rolling30 = np.mean(
                history[-30:]
            )

        return pd.DataFrame(predictions)

    # ==================================================
    # FORECAST BUTTON
    # ==================================================

    if st.button(
        "Forecast Demand",
        key="forecast_btn"
    ):

        result = forecast_demand(
            selected_speciality,
            forecast_days,
            selected_date
        )

        st.subheader(
            "Forecast Results"
        )

        st.dataframe(result)

        result["Date"] = pd.to_datetime(
            result["Date"]
        ).dt.date

        fig = px.bar(
            result,
            x="Date",
            y="Predicted Appointments",
            title=f"{selected_speciality} Demand Forecast"
        )

        fig.update_xaxes(
            type="category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # -----------------------------------
        # Actual vs Predicted Validation
        # -----------------------------------

        selected_date_ts = pd.Timestamp(
            selected_date
        )

        actual_row = comparison_df[
            comparison_df["Date"]
            == selected_date_ts
        ]

        if not actual_row.empty:

            st.subheader(
                "Selected Date Validation"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Actual Appointments",
                int(
                    actual_row[
                        "Actual"
                    ].iloc[0]
                )
            )

            c2.metric(
                "Predicted Appointments",
                int(
                    actual_row[
                        "Predicted"
                    ].iloc[0]
                )
            )

    # ==================================================
    # MODEL PERFORMANCE
    # ==================================================

    st.subheader(
        "📊 Model Performance"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", "5.74")
    c2.metric("MAE", "3.42")
    c3.metric("R²", "0.9995")
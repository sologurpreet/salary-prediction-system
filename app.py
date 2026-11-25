import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODEL

pipeline = joblib.load("salary_prediction_pipeline.pkl")

# LOAD DATA

df = pd.read_csv("expected_ctc.csv")
df = df.fillna("Unknown")

# HELPER

def get_options(col):
    return sorted(df[col].astype(str).unique())

# CUSTOM CSS

st.markdown("""
<style>

.main{
    background:#f5f7fb;
}

.block-container{
    padding-top:1rem;
}

.dashboard-title{
    font-size:42px;
    font-weight:bold;
    color:white;
}

.dashboard-subtitle{
    color:white;
    font-size:18px;
}

.header-box{
    padding:30px;
    border-radius:18px;
    background:linear-gradient(90deg,#0f62fe,#4589ff);
    margin-bottom:25px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,.1);
}

</style>
""", unsafe_allow_html=True)

# HEADER

st.markdown("""
<div class="header-box">

<div class="dashboard-title">
💼 HR Analytics Dashboard
</div>

<div class="dashboard-subtitle">
Machine Learning Based Expected Annual CTC Prediction System
</div>

</div>
""", unsafe_allow_html=True)

# SIDEBAR

st.sidebar.image(
    "https://img.icons8.com/color/96/businessman.png",
    width=90
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select",
    [
        "Salary Prediction",
        "Dataset Overview",
        "About Project"
    ]
)

if page != "Salary Prediction":

    if page == "Dataset Overview":

        st.title("📊 Dataset Overview")

        st.write(df.head())

        st.write("Rows:", df.shape[0])

        st.write("Columns:", df.shape[1])

        st.bar_chart(df["Department"].value_counts().head(10))

    else:

        st.title("About Project")

        st.write("""
This application predicts the Expected Annual CTC of a candidate
using a Random Forest Machine Learning Pipeline.
""")

    st.stop()

# KPI CARDS

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("📊 Dataset", f"{len(df):,} Records"),
    ("🧠 Model", "Random Forest"),
    ("🎯 Target", "Expected CTC"),
    ("📁 Features", "26 Input Features")
]

for col, (title, value) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(f"""
        <div style="
            background:white;
            padding:20px;
            border-radius:15px;
            box-shadow:0 3px 10px rgba(0,0,0,.12);
            text-align:center;
            min-height:120px;
        ">
            <h4 style="margin-bottom:10px;">{title}</h4>
            <h2 style="color:#0f62fe;">{value}</h2>
        </div>
        """, unsafe_allow_html=True)

# INPUT FORM

st.header("👤 Candidate Information")

with st.form("prediction_form"):

    tab1, tab2 = st.columns(2)

    # LEFT COLUMN

    with tab1:

        st.subheader("💼 Experience")

        total_experience = st.number_input(
            "Total Experience (Years)",
            0,
            40,
            5
        )

        field_experience = st.number_input(
            "Experience in Applied Field",
            0,
            40,
            4
        )

        companies = st.number_input(
            "Companies Worked",
            0,
            20,
            1
        )

        st.subheader("🏢 Employment")

        department = st.selectbox(
            "Department",
            get_options("Department")
        )

        role = st.selectbox(
            "Role",
            get_options("Role")
        )

        industry = st.selectbox(
            "Industry",
            get_options("Industry")
        )

        organization = st.selectbox(
            "Organization",
            get_options("Organization")
        )

        designation = st.selectbox(
            "Designation",
            get_options("Designation")
        )

        current_ctc = st.number_input(
            "Current Annual CTC (₹)",
            0,
            10000000,
            500000,
            step=10000
        )

        inhand_offer = st.number_input(
            "Current In-hand Offer (₹)",
            0,
            10000000,
            500000,
            step=10000
        )

    # RIGHT COLUMN

    with tab2:

        st.subheader("🎓 Education")

        education = st.selectbox(
            "Education",
            get_options("Education")
        )

        graduation_specialization = st.selectbox(
            "Graduation Specialization",
            get_options("Graduation_Specialization")
        )

        university_grad = st.selectbox(
            "Graduation University",
            get_options("University_Grad")
        )

        passing_grad = st.number_input(
            "Graduation Year",
            1980,
            2035,
            2020
        )

        pg_specialization = st.selectbox(
            "PG Specialization",
            get_options("PG_Specialization")
        )

        university_pg = st.selectbox(
            "PG University",
            get_options("University_PG")
        )

        passing_pg = st.number_input(
            "PG Passing Year",
            1980,
            2035,
            2022
        )

        phd_specialization = st.selectbox(
            "PhD Specialization",
            get_options("PHD_Specialization")
        )

        university_phd = st.selectbox(
            "PhD University",
            get_options("University_PHD")
        )

        passing_phd = st.number_input(
            "PhD Passing Year",
            1980,
            2035,
            2024
        )

        current_location = st.selectbox(
            "Current Location",
            get_options("Curent_Location")
        )

        preferred_location = st.selectbox(
            "Preferred Location",
            get_options("Preferred_location")
        )

        appraisal = st.selectbox(
            "Last Appraisal Rating",
            get_options("Last_Appraisal_Rating")
        )

        publications = st.number_input(
            "Publications",
            0,
            100,
            0
        )

        certifications = st.number_input(
            "Certifications",
            0,
            100,
            0
        )

        international_degree = st.selectbox(
            "International Degree",
            get_options("International_degree_any")
        )

    st.markdown("---")

    predict_button = st.form_submit_button(
        "🚀 Predict Expected Annual CTC",
        use_container_width=True
    )
# PREDICTION

if predict_button:

    # Create input dataframe
    input_df = pd.DataFrame({

        "Total_Experience":[total_experience],
        "Total_Experience_in_field_applied":[field_experience],
        "Department":[department],
        "Role":[role],
        "Industry":[industry],
        "Organization":[organization],
        "Designation":[designation],
        "Education":[education],
        "Graduation_Specialization":[graduation_specialization],
        "University_Grad":[university_grad],
        "Passing_Year_Of_Graduation":[passing_grad],
        "PG_Specialization":[pg_specialization],
        "University_PG":[university_pg],
        "Passing_Year_Of_PG":[passing_pg],
        "PHD_Specialization":[phd_specialization],
        "University_PHD":[university_phd],
        "Passing_Year_Of_PHD":[passing_phd],
        "Curent_Location":[current_location],
        "Preferred_location":[preferred_location],
        "Current_CTC":[current_ctc],
        "Inhand_Offer":[inhand_offer],
        "Last_Appraisal_Rating":[appraisal],
        "No_Of_Companies_worked":[companies],
        "Number_of_Publications":[publications],
        "Certifications":[certifications],
        "International_degree_any":[international_degree]

    })

    # Progress animation
    progress = st.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    prediction = pipeline.predict(input_df)

    predicted_salary = float(prediction[0])

    salary_increase = predicted_salary - current_ctc

    if current_ctc > 0:
        increase_percent = (salary_increase/current_ctc)*100
    else:
        increase_percent = 0

    st.success("✅ Prediction Completed Successfully!")

    st.markdown("## 💰 Prediction Result")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric(
            "Current Annual CTC",
            f"₹ {current_ctc:,.0f}"
        )

    with c2:
        st.metric(
            "Predicted Annual CTC",
            f"₹ {predicted_salary:,.0f}"
        )

    with c3:
        st.metric(
            "Salary Difference",
            f"₹ {salary_increase:,.0f}",
            f"{increase_percent:.1f}%"
        )

    st.markdown("---")
    
# SALARY COMPARISON

    comparison = pd.DataFrame({

        "Type":[
            "Current Salary",
            "Predicted Salary"
        ],

        "Salary":[
            current_ctc,
            predicted_salary
        ]

    })

    fig = px.bar(

        comparison,

        x="Type",

        y="Salary",

        text="Salary",

        title="Salary Comparison"

    )

    fig.update_traces(texttemplate='₹%{text:,.0f}')

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# SUMMARY

    st.markdown("## 📋 Candidate Summary")

    summary = pd.DataFrame({

        "Field":[

            "Experience",

            "Department",

            "Role",

            "Education",

            "Current Salary",

            "Predicted Salary"

        ],

        "Value":[

            f"{total_experience} Years",

            department,

            role,

            education,

            f"₹ {current_ctc:,.0f}",

            f"₹ {predicted_salary:,.0f}"

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )
# DOWNLOAD CSV

    csv = summary.to_csv(index=False)

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        file_name="salary_prediction_report.csv",

        mime="text/csv"

    )
    st.markdown("---")

st.markdown("""
<div style='text-align:center; padding:20px; color:gray;'>

<b>Salary Prediction System</b><br>

Developed using Streamlit • Scikit-Learn • Random Forest Regression

<br><br>

© 2026 Gurpreet Singh

</div>
""", unsafe_allow_html=True)
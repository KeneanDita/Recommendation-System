import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

teacher_vectors = np.load("assets/teacher_vectors.npy", allow_pickle=True)
scaler = joblib.load("assets/scaler.pkl")
features = joblib.load("assets/features.pkl")
data = pd.read_csv("assets/original_teachers_df.csv")

primary_subjects = sorted({col.split('_')[-1] for col in features if "Primary_Subject_" in col})
secondary_subjects = sorted({col.split('_')[-1] for col in features if "Secondary_Subject_" in col})
education_levels = sorted({col.split('_')[-1] for col in features if "Education_Level_" in col})
teaching_styles = sorted({col.split('_')[-1] for col in features if "Teaching_Style_" in col})
certifications = sorted({col.split('_')[-1] for col in features if "Certifications_" in col})
availabilities = sorted({col.split('_')[-1] for col in features if "Availability_" in col})
languages = sorted({col.split('_')[-1] for col in features if "Language_" in col})
genders = ["Male", "Female", "Non-Binary"]
countries = ["Canada", "Australia", "Germany", "UK", "India", "USA", "France"]

st.set_page_config(page_title="Teacher Recommendation System", layout="wide")
st.title("Teacher Recommendation System")

st.sidebar.header("Filter Options")
primary_subject = st.sidebar.selectbox("Primary Subject", primary_subjects)
secondary_subject = st.sidebar.selectbox("Secondary Subject", secondary_subjects)
education = st.sidebar.selectbox("Education Level", education_levels)
teaching_style = st.sidebar.selectbox("Teaching Style", teaching_styles)
certification = st.sidebar.selectbox("Certification", certifications)
availability = st.sidebar.selectbox("Availability", availabilities)
language = st.sidebar.selectbox("Language", languages)
gender = st.sidebar.selectbox("Gender", genders)
country = st.sidebar.selectbox("Country", countries)
experience = st.sidebar.slider("Years of Experience", 1, 30, 10)
rating = st.sidebar.slider("Minimum Rating", 3.0, 5.0, 4.5)
courses = st.sidebar.slider("Courses Taught", 5, 50, 20)
research_active = st.sidebar.checkbox("Research Active", value=True)

input_data = pd.DataFrame([{
    "Years_of_Experience": experience,
    "Student_Rating": rating,
    "Courses_Taught": courses
}])
scaled_vals = scaler.transform(input_data)

course_profile = {
    f"Primary_Subject_{primary_subject}": 1,
    f"Secondary_Subject_{secondary_subject}": 1,
    f"Education_Level_{education}": 1,
    f"Teaching_Style_{teaching_style}": 1,
    f"Certifications_{certification}": 1,
    f"Availability_{availability}": 1,
    f"Language_{language}": 1,
    f"Gender_{gender}": 1,
    f"Country_{country}": 1,
    "Years_of_Experience": experience,
    "Student_Rating": rating,
    "Courses_Taught": courses,

    "Is_Research_Active": 1 if research_active else 0
}


query_vector = np.array([course_profile.get(f, 0) for f in features]).reshape(1, -1)

similarities = cosine_similarity(query_vector, teacher_vectors)
top_indices = similarities[0].argsort()[-10:][::-1]

st.header("Top Recommended Teachers")

columns_to_display = [
    "Teacher_ID", "Full_Name", "Email", "Primary_Subject", 
    "Student_Rating", "Years_of_Experience", "Courses_Taught"
]

available_columns = [col for col in columns_to_display if col in data.columns]
recommended_teachers = data.iloc[top_indices][available_columns].reset_index(drop=True)

st.dataframe(recommended_teachers.style.format({
    "Student_Rating": "{:.2f}",
    "Years_of_Experience": "{:.1f}",
    "Courses_Taught": "{:.0f}"
}))

# Teacher Recommendation System

This project is a teacher recommendation system that helps users find the most suitable teachers based on various factors, including experience, rating, courses taught, and other attributes. It consists of two main components:

1. **Data Processing Notebook** - Prepares and processes the teacher data, including scaling numerical features and encoding categorical features.
2. **Streamlit UI** - Provides a user-friendly interface for searching and recommending teachers.

## Key Features

* Recommends teachers based on experience, rating, courses taught, and other relevant attributes.
* Real-time filtering using various criteria such as subjects, education level, certifications, and teaching styles.
* Scalable design to support large datasets.

## Setup Instructions

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset:**

   * Ensure your dataset is saved as `synthetic_teachers_dataset.csv` in the assets directory.

3. **Run the Data Processing Notebook:**

   * Open and run `Teachers_Recommendation_Updated.ipynb` to generate the encoded teacher vectors and scaler files.

4. **Run the Streamlit App:**

   ```bash
   streamlit run streamlit.py
   ```

## Directory Structure

```
├── assets
│   ├── scaler_updated.pkl
|   ├── synthetic_teachers_dataset.csv
│   ├── features_updated.pkl
│   └── teacher_vectors_updated.npy
├── Teachers_Recommendation_Updated.ipynb
├── streamlit.py
└── README.md
```

## Future Improvements

* Add personalized recommendation based on student feedback.
* Implement a machine learning model for better prediction accuracy.
* Integrate a database for real-time updates.

### If you want to explore the docker image

[Docker link](https://hub.docker.com/repository/docker/keneandita/recommendation_system/general)

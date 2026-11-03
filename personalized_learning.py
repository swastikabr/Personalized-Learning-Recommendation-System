import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


# 1. FOLDERS
os.makedirs("outputs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
print(" PERSONALIZED LEARNING RECOMMENDATION ")

# 2. CREATE STUDENT DATASET
np.random.seed(42)

n = 500

data = pd.DataFrame({
    "previous_score": np.random.randint(30, 101, n),
    "quiz_score": np.random.randint(20, 101, n),
    "study_hours": np.random.randint(1, 15, n),

    "interest": np.random.choice(
        ["Python", "Data Science", "Web Development",
         "Machine Learning", "Database"], n),

    "learning_style": np.random.choice(
        ["Visual", "Practical", "Reading", "Auditory"], n),

    "completed_topics": np.random.choice(
        ["Basics", "Intermediate", "Advanced"], n),

    "difficulty": np.random.choice(
        ["Beginner", "Intermediate", "Advanced"], n),

    "study_consistency": np.random.choice(
        ["Regular", "Irregular"], n)
})


# 3. GENERATE RECOMMENDATION LABELS
def recommend_resource(row):

    if row["previous_score"] < 40:
        return "Beginner Tutorial"

    elif row["interest"] == "Python":
        return "Python Practice"

    elif row["interest"] == "Data Science":
        return "Data Visualization Practice"

    elif row["interest"] == "Web Development":
        return "Web Development Project"

    elif row["interest"] == "Machine Learning":
        return "Machine Learning Tutorial"

    else:
        return "SQL Practice"


data["recommended_resource"] = data.apply(
    recommend_resource, axis=1
)


# 4. DATA CLEANING
print("\nDATASET INFORMATION")

print("Dataset Shape:", data.shape)
print("Missing Values:", data.isnull().sum().sum())
print("Duplicate Rows:", data.duplicated().sum())

data = data.drop_duplicates()
data = data.dropna()

data.to_csv("outputs/student_learning_dataset.csv", index=False)

print("\nDataset saved successfully!")


# 5. EXPLORATORY DATA ANALYSIS
print("\nDATASET SUMMARY")
print(data.describe())

# Graph 1: Previous Score Distribution
plt.figure(figsize=(7, 5))
plt.hist(data["previous_score"], bins=10, edgecolor="black")
plt.title("Student Previous Score Distribution")
plt.xlabel("Previous Score")
plt.ylabel("Number of Students")
plt.savefig("graphs/score_distribution.png")
plt.show()

# Graph 2: Study Hours vs Quiz Score
plt.figure(figsize=(7, 5))
plt.scatter(data["study_hours"], data["quiz_score"])
plt.title("Study Hours vs Quiz Score")
plt.xlabel("Study Hours")
plt.ylabel("Quiz Score")
plt.savefig("graphs/study_hours_vs_score.png")
plt.show()

# Graph 3: Learning Style Distribution
plt.figure(figsize=(7, 5))
data["learning_style"].value_counts().plot(kind="bar")
plt.title("Student Learning Styles")
plt.xlabel("Learning Style")
plt.ylabel("Number of Students")
plt.savefig("graphs/learning_style.png")
plt.show()

# Graph 4: Student Interests
plt.figure(figsize=(7, 5))
data["interest"].value_counts().plot(kind="bar")
plt.title("Student Interest Distribution")
plt.xlabel("Interest")
plt.ylabel("Number of Students")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("graphs/student_interests.png")
plt.show()


# 6. PREPARE DATA FOR MACHINE LEARNING
X = data.drop("recommended_resource", axis=1)
y = data["recommended_resource"]

numeric_features = [
    "previous_score",
    "quiz_score",
    "study_hours"
]

categorical_features = [
    "interest",
    "learning_style",
    "completed_topics",
    "difficulty",
    "study_consistency"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)


# 7. TRAIN MACHINE LEARNING MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

print("\nTraining Machine Learning Model...")
model.fit(X_train, y_train)
print("Model Training Completed!")


# 8. MODEL EVALUATION
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, predictions, zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title("Model Confusion Matrix")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")
plt.show()


# 9. FEATURE IMPORTANCE
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = model.named_steps[
    "classifier"
].feature_importances_

feature_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_data = feature_data.sort_values(
    by="Importance", ascending=False
)

feature_data.to_csv(
    "outputs/feature_importance.csv", index=False
)

plt.figure(figsize=(9, 6))

top_features = feature_data.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("graphs/feature_importance.png")
plt.show()


# 10. SAVE TRAINED MODEL
joblib.dump(model, "outputs/personalized_learning_model.pkl")
print("\nTrained model saved successfully!")


# 11. PERSONALIZED RECOMMENDATION
print("\n========================================")
print(" ENTER STUDENT DETAILS")
print("========================================")

student = pd.DataFrame([{
    "previous_score": float(input("Previous Score (0-100): ")),
    "quiz_score": float(input("Quiz Score (0-100): ")),
    "study_hours": float(input("Study Hours per Week: ")),

    "interest": input(
        "Interest (Python/Data Science/Web Development/"
        "Machine Learning/Database): "
    ),

    "learning_style": input(
        "Learning Style (Visual/Practical/Reading/Auditory): "
    ),

    "completed_topics": input(
        "Completed Topics (Basics/Intermediate/Advanced): "
    ),

    "difficulty": input(
        "Difficulty (Beginner/Intermediate/Advanced): "
    ),

    "study_consistency": input(
        "Study Consistency (Regular/Irregular): "
    )
}])

recommendation = model.predict(student)[0]

print("\n========================================")
print(" PERSONALIZED LEARNING RECOMMENDATION")
print("========================================")
print("Recommended Resource:", recommendation)
print("\nRecommendation Generated Successfully!")
print("\nAll project files are saved in the outputs and graphs folders.")

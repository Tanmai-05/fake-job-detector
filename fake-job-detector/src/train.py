from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib

from preprocess import load_data, get_features_and_labels

print("🔍 Loading data...")
df = load_data()

print("✅ Splitting data...")
X_train, X_test, y_train, y_test = get_features_and_labels(df)

print("🧠 Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)

print("📊 Evaluating model...")
print(classification_report(y_test, pipeline.predict(X_test)))

print("💾 Saving model...")
joblib.dump(pipeline, 'models/job_detector.pkl')

print("✅ Model training complete!")

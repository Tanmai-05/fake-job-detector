🕵️‍♂️ Fake Job Detector using BERT

This project uses **NLP with BERT** (Bidirectional Encoder Representations from Transformers) to classify job postings as **real** or **fake**. It helps job seekers avoid scams by analyzing job descriptions and titles.

🚀 Project Highlights

- ✅ Built with **DistilBERT** for efficiency and accuracy  
- 🧠 Trained on real-world dataset of job postings  
- 📊 Visualizes results using **Streamlit Web App**  
- 📁 Organized with clean modular code in `src/` directory  
- 🧪 Predicts fake or real job based on user input


 📂 Project Structure

```bash
fake-job-detector/
├── app.py                     # Streamlit app
├── bert_train.py              # Model training script
├── bert_predict.py            # Prediction interface (if separate)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── fake_job_postings.csv  # Dataset
├── models/
│   └── job_detector.pkl       # Trained model file
├── src/
│   ├── preprocess.py          # Preprocessing logic
│   ├── train.py               # Model training utilities
│   └── predict.py             # Prediction utilities
````

 🛠️ How to Run the App

1. Clone the Repository


git clone https://github.com/Tanmai-05/fake-job-detector.git
cd fake-job-detector

2. Install Dependencies

Make sure Python 3.8+ is installed.

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run app.py

🧪 Example Inputs

* **Job Title:** `Data Entry Job – Earn ₹5000 daily`
* **Job Description:** `No experience needed. Just type simple words. Contact us on WhatsApp.`

Expected Prediction: ❌ Fake

 🧠 Model Details

* Model: `DistilBERTForSequenceClassification`
* Fine-tuned on fake job postings dataset
* Accuracy: \~98% on validation data
* Tokenizer: `distilbert-base-uncased`

---

Dataset used
Dataset Source: [Fake Job Postings Dataset on Kaggle](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction)

---

📎 License

This project is open-source and available under the [MIT License](LICENSE).

---

 👨‍💻 Author

**Tanmai Manda**
GitHub: [@Tanmai-05](https://github.com/Tanmai-05)


## ⭐️ Give a Star

If you like this project, consider giving it a ⭐️ on GitHub!



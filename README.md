# Salary Prediction App

This is a Flask web application that predicts employee salaries using a **Ridge Regression model** trained on multinational workforce data.
The app supports:

* **Single Prediction** – Enter job features (Region, Department, Job Industry, Work Mode, Experience, Performance Rating) to get a salary prediction.
* **Batch Prediction** – Upload a CSV file with multiple records to generate salary predictions for all entries.
* **Summary Statistics** – View average and total predicted salaries for uploaded datasets.

---

## 🔧 Project Structure

```
.
├── app.py                  # Main Flask application
├── ridge_pipeline.pkl      # Saved Ridge Regression model + preprocessor
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # UI for single & batch prediction
│   └── batch_result.html   # Results page for batch predictions
└── README.md               # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/ProfFausat/salary-predictor.git
   cd salary-predictor
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app**

   ```bash
   python app.py
   ```

5. **Open in browser**
   Visit: `http://127.0.0.1:5000`

---

## 📂 Batch Prediction CSV Format

The CSV must include the following columns (same as training features):

* `Region_Located`
* `Job_Industry`
* `Work_Mode`
* `Department`
* `Experience_Years`
* `Performance_Rating`

### Example CSV

```csv
Region_Located,Job_Industry,Work_Mode,Department,Experience_Years,Performance_Rating
Europe,Technology / IT,Remote,Engineering,5,4
Asia,Finance / Accounting,On-site,Finance,8,3
North America,Marketing / Branding,Remote,Marketing,3,5
```

---

## 📊 Model Information

* Algorithm: **Ridge Regression**
* Preprocessing: Encodes categorical variables + scales numeric features.
* Target variable: `Log_Salary_INR`

---

## ✨ Future Enhancements

* Support additional job features.
* Deploy with CI/CD pipeline on AWS.

from flask import Flask, request, render_template, send_file
import pickle
import numpy as np
import pandas as pd
import io

# sklearn pieces we will inspect to try to extract categories
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ---------------- Load Ridge model + preprocessor ----------------
with open("ridge_pipeline.pkl", "rb") as f:
    model_objects = pickle.load(f)

preprocessor = model_objects["preprocessor"]
ridge_model = model_objects["ridge_model"]

# ---------------- Static option lists (from your message) ----------------
REGION_OPTIONS = [
    "Europe", "North America", "Asia", "Unknown", "Africa",
    "Oceania", "South America", "Antarctica"
]

JOB_INDUSTRY_OPTIONS = [
    "Technology / IT",
    "Marketing / Branding",
    "Human Resources",
    "Operations / Logistics / Supply Chain",
    "Finance / Accounting",
    "Sales / Business Development",
    "Research / Science"
]

WORK_MODE_OPTIONS = ["On-site", "Remote"]

# ---------------- Helper: try to extract categories for 'Department' from preprocessor ----------------
def _get_categories_for_column(preproc, column_name):
    """
    Attempt to read categories for 'column_name' from a ColumnTransformer-style preprocessor.
    Returns list of categories if found, otherwise None.
    """
    try:
        # ColumnTransformer exposes .transformers_
        for name, transformer, cols in preproc.transformers_:
            # normalize cols to a list
            if isinstance(cols, (list, tuple, np.ndarray)):
                cols_list = list(cols)
            else:
                cols_list = [cols]

            if column_name in cols_list:
                # If transformer is a Pipeline, search for OneHotEncoder inside
                enc = None
                if isinstance(transformer, Pipeline):
                    for _, step in transformer.steps:
                        if isinstance(step, OneHotEncoder):
                            enc = step
                            break
                        # sometimes steps might be named differently; above handles typical case
                else:
                    # transformer might be OneHotEncoder itself
                    if isinstance(transformer, OneHotEncoder):
                        enc = transformer
                    # or it might have named_steps (another pipeline-like object)
                    elif hasattr(transformer, "named_steps"):
                        for s in transformer.named_steps.values():
                            if isinstance(s, OneHotEncoder):
                                enc = s
                                break

                if enc is not None and hasattr(enc, "categories_"):
                    # categories_ is a list; find index of our column in cols_list
                    idx = cols_list.index(column_name)
                    # some encoders hold categories_ as list of arrays per transformed column
                    cats = list(enc.categories_[idx])
                    return cats
    except Exception:
        # safe fallback to returning None if anything goes wrong
        return None

    return None

DEPARTMENT_OPTIONS = _get_categories_for_column(preprocessor, "Department")
if not DEPARTMENT_OPTIONS:
    # fallback default — replace with your exact list if you have it
    DEPARTMENT_OPTIONS = ["HR", "Finance", "IT", "Operations", "Sales", "Marketing"]

# Print to console when server boots so you can confirm extraction
print("Loaded Department options:", DEPARTMENT_OPTIONS)

# In-memory storage for last batch CSV (for download)
last_output = None
last_df_summary = None


# small helper to render the index page always with dropdown lists available
def render_home(prediction=None, error_msg=None):
    return render_template(
        "index.html",
        prediction=prediction,
        error_msg=error_msg,
        region_options=REGION_OPTIONS,
        job_industry_options=JOB_INDUSTRY_OPTIONS,
        work_mode_options=WORK_MODE_OPTIONS,
        department_options=DEPARTMENT_OPTIONS
    )


# ---------------- Home (single prediction) ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error_msg = None

    if request.method == "POST" and "predict_single" in request.form:
        # Gather inputs
        region = request.form.get("Region_Located", "").strip()
        industry = request.form.get("Job_Industry", "").strip()
        work_mode = request.form.get("Work_Mode", "").strip()
        department = request.form.get("Department", "").strip()

        try:
            exp_years = int(float(request.form.get("Experience_Years", "").strip()))
            perf_rating = int(float(request.form.get("Performance_Rating", "").strip()))
        except Exception:
            return render_home(prediction=None, error_msg="Experience_Years and Performance_Rating must be integers.")

        # Validate required fields
        if not all([region, industry, work_mode, department]):
            return render_home(prediction=None, error_msg="Please select/enter all required fields.")

        # Validate ranges
        if (exp_years <= 0):
            return render_home(prediction=None, error_msg="Experience_Years must be greater than 0.")
        if not (1 <= perf_rating <= 5):
            return render_home(prediction=None, error_msg="Performance_Rating must be between 1 and 5.")

        # Create single-row dataframe (order of columns matches training)
        input_df = pd.DataFrame([{
            "Region_Located": region,
            "Job_Industry": industry,
            "Work_Mode": work_mode,
            "Department": department,
            "Experience_Years": exp_years,
            "Performance_Rating": perf_rating
        }])

        # Predict (preprocessor handles categorical encoding)
        try:
            X_transformed = preprocessor.transform(input_df)
            prediction = ridge_model.predict(X_transformed)[0]  # this is log salary as requested
            prediction = np.e**prediction
        except Exception as e:
            return render_home(prediction=None, error_msg=f"Prediction error: {str(e)}")

    return render_home(prediction=prediction, error_msg=error_msg)


# ---------------- Batch prediction ----------------
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    global last_output, last_df_summary

    file = request.files.get("file")
    if not file:
        return render_home(prediction=None, error_msg="No file uploaded. Please choose a CSV file.")

    try:
        df = pd.read_csv(file)
    except Exception:
        return render_home(prediction=None, error_msg="Unable to read CSV file. Ensure it's a valid CSV.")

    required_cols = ["Region_Located", "Job_Industry", "Work_Mode", "Department", "Experience_Years", "Performance_Rating"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return render_home(prediction=None, error_msg=f"CSV missing columns: {', '.join(missing)}. Required: {', '.join(required_cols)}")

    # Ensure numeric columns are integers within expected range
    try:
        df["Experience_Years"] = pd.to_numeric(df["Experience_Years"]).astype(int)
        df["Performance_Rating"] = pd.to_numeric(df["Performance_Rating"]).astype(int)
    except Exception:
        return render_home(prediction=None, error_msg="Experience_Years and Performance_Rating must be numeric integers in the CSV.")

    # Validate ranges (optionally you can drop or flag invalid rows; here we reject)
    if df["Experience_Years"].lt(0).any() or df["Experience_Years"].gt(15).any():
        return render_home(prediction=None, error_msg="All Experience_Years values must be between 0 and 15.")
    if df["Performance_Rating"].lt(1).any() or df["Performance_Rating"].gt(5).any():
        return render_home(prediction=None, error_msg="All Performance_Rating values must be between 1 and 5.")

    # Predictions
    try:
        X_transformed = preprocessor.transform(df[required_cols])
        preds = ridge_model.predict(X_transformed)  # log salaries
    except Exception as e:
        return render_home(prediction=None, error_msg=f"Prediction error: {str(e)}")

    df["Predicted_Log_Salary"] = preds

    # Save CSV in memory for download
    output = io.BytesIO()
    df.to_csv(output, index=False)
    last_output = output.getvalue()
    output.seek(0)

    # Summary
    total_rows = len(df)
    mean_pred = float(df["Predicted_Log_Salary"].mean()) if total_rows > 0 else 0.0
    sum_pred = float(df["Predicted_Log_Salary"].sum()) if total_rows > 0 else 0.0
    last_df_summary = {"rows": total_rows, "mean": mean_pred, "sum": sum_pred}

    # Convert DF to HTML (Bootstrap)
    table_html = df.to_html(classes="table table-striped table-bordered", index=False, escape=False)

    return render_template("batch_result.html", table_html=table_html, summary=last_df_summary)


# ---------------- Download last batch CSV ----------------
@app.route("/download_csv")
def download_csv():
    global last_output
    if not last_output:
        return "No batch results available. Upload a CSV and run batch predictions first.", 400

    buffer = io.BytesIO(last_output)
    buffer.seek(0)
    return send_file(buffer, mimetype="text/csv", as_attachment=True, download_name="predicted_log_salary.csv")


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)

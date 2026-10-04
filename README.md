# Global Macroeconomic GDP Forecaster 🌍📈

Welcome to the **Global Macroeconomic GDP Forecaster**! This project is an end-to-end data pipeline and interactive web application that predicts a country's future GDP growth based on key economic factors.

## 🧠 What is this project?
In simple terms, this tool acts as a crystal ball for the economy. It looks at a country's historical financial health—like how much debt they have, inflation rates, and past GDP numbers—and uses Machine Learning algorithms to guess where their GDP is heading in the coming years.

### Key Features
* **Interactive Dashboard:** A beautiful, easy-to-use web interface built with Streamlit.
* **Smart Predictions:** Uses two different AI models (Ridge Regression and Gradient Boosting) to forecast future economic trends.
* **Data Cleaning & Processing:** Automatically handles missing data and prepares raw numbers for the AI.
* **Automated Reports:** Generates Excel reports for hypothesis testing, model evaluation, and business interpretation.

---

## 🛠️ How it Works (Under the Hood)
1. **Data Ingestion:** We start with real-world economic data (`Data Realistic.xlsx` and `uncleaned.csv`).
2. **Cleaning Pipeline:** Our processing scripts (`cleaner.py`, `CLEAN8.PY`, etc.) scrub the data—removing errors and filling in blanks.
3. **Training the AI:** The clean data is fed to our models (`model.py`). The models "learn" the patterns of how inflation and debt affect GDP. They are then saved as `.pkl` files so they don't have to relearn from scratch every time.
4. **The Web App:** The user interface (`app.py`) loads these pre-trained models and displays the interactive charts and predictions to the user.

---

## 📂 Project Structure
All the important files are stored in this main directory:
* **`app.py`** - The main user interface file (run this to launch the app).
* **`.py` scripts** - Background processors that handle data cleaning, interpretations, and model building (e.g., `cleaner.py`, `model.py`, `exp10_business_analytics.py`).
* **`.xlsx` & `.csv` files** - The databases! Contains both raw data and generated output reports.
* **`.pkl` files** - The "brains" of the operation. These are saved Machine Learning models and data scalers.
* **`.gitignore`** - Tells GitHub which background files to ignore (like cache files).

---

## 🚀 How to Run the App on Your Computer

1. **Open your terminal (PowerShell or Command Prompt).**
2. **Navigate to this folder:**
   ```bash
   cd "C:\Users\asmit\Downloads\TEST\Project_Code_DB"
   ```
3. **Install the required tools:**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn openpyxl
   ```
4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
5. The application will automatically open in your web browser!

---

## 📊 Outputs & Reports
When the data processing pipeline runs, it generates several highly detailed Excel reports. You can view these directly in the folder:
- *Business_Interpretation_Report.xlsx*
- *Hypothesis Testing Report.xlsx*
- *Master_Model_Evaluation_Report.xlsx*

Enjoy exploring the macroeconomic trends!

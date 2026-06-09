# 🎓 University Management Predictive Analytics Dashboard

> Interactive 10-step Streamlit analytics dashboard for university student data — validation, profiling, GPA analytics, attendance heatmaps, predictive modeling prep, and automated report generation.

---

## 🗂️ Project Structure

```
UniversityDashboard/
├── app.py                                          # Full ML pipeline (Regression + Classification + Apriori)
├── DashBoard.py                                    # 10-step interactive Streamlit dashboard
├── University_Management_Cleaned.xlsx              # Cleaned dataset (students, courses, enrollments, grades)
├── University_Management_Processed_With_Validation.xlsx  # Validated & processed dataset
└── Predictive_Analytics_Tasks.xlsx                 # ML task definitions and results
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly Express |
| Machine Learning | Scikit-learn (Random Forest, XGBoost, Decision Tree, Ridge, Lasso) |
| Association Mining | mlxtend (Apriori) |
| Reporting | FPDF, xlsxwriter, SMTP email |

---

## 📋 10-Step Dashboard (`DashBoard.py`)

| Step | Feature |
|------|---------|
| 1 | Excel file upload — loads 4 sheets: students, courses, enrollments, grades |
| 2 | Validation & Quality Checks — unique ID counts, enrollment trends, grade distribution, dropped students analysis |
| 3 | Data Dictionary — auto-generated column metadata with types and example values |
| 4 | Profiling Summary — row/column counts, null value report per table |
| 5 | Transformation Log — documents all data cleaning steps applied |
| 6 | Attendance vs GPA Heatmap — density heatmap showing attendance-performance correlation |
| 7 | Top Students & Courses — top 10 students by GPA, top 10 courses by average GPA |
| 8 | Predictive Analytics Prep — defines regression features, target (GPA), and evaluation metrics |
| 9 | Download Cleaned Data — exports all 4 sheets as a single cleaned Excel file |
| 10 | HTML Report Generation — auto-generates downloadable summary report |

---

## 🤖 ML Pipeline (`app.py`)

### Regression — Predict GPA
- Models: Linear Regression, Ridge, Lasso
- Features: attendance rate, credits, student age, department, semester
- Metrics: MAE, RMSE, R²
- Outputs: Actual vs Predicted scatter, residual distribution, feature importance

### Classification — Dropout Prediction
- Models: Decision Tree, Random Forest, XGBoost / GradientBoosting
- Features: attendance rate, past grade avg, courses dropped, semester load
- Metrics: Accuracy, Precision, Recall, AUC
- Outputs: Confusion matrix, feature importance, high-risk student list (downloadable)

### Association Rule Mining — Subject Co-failure Patterns
- Algorithm: Apriori (mlxtend)
- Configurable support/confidence thresholds
- Output: Lift heatmap, ranked association rules table

### Model Comparison Panel
- Side-by-side accuracy/precision/recall comparison across classifiers

### High-Risk Dashboard
- Random Forest risk scoring on all students
- Per-student justification (top contributing features)
- Downloadable high-risk Excel list with threshold slider

### Reports & Email
- Automated PDF report via FPDF
- SMTP email dispatch with PDF + high-risk Excel attachment

---

## 📊 Visualizations

- Enrollment trend line chart (students per semester)
- Grade distribution bar chart
- Dropped students per semester
- Attendance vs GPA density heatmap
- Top 10 students and courses by GPA
- Correlation heatmap (interactive Plotly)
- Sunburst — Department → Semester → Avg GPA
- Radar chart — normalized feature comparison
- Actual vs Predicted scatter (regression)
- Residual distribution histogram
- Confusion matrix heatmap
- Feature importance bar charts

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SanketSutar075/University-Management-Dashboard.git
cd University-Management-Dashboard
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn mlxtend xgboost fpdf xlsxwriter openpyxl
```

### 3. Run the 10-step dashboard
```bash
streamlit run DashBoard.py
```

### 4. Run the full ML pipeline
```bash
streamlit run app.py
```

### 5. Upload the dataset
Upload `University_Management_Cleaned.xlsx` when prompted — it contains 4 sheets:
- `students`
- `courses`
- `enrollments`
- `enrollments_with_grades`

---

## 📈 Key Results

- **10-step** structured analytics workflow from raw data to report
- **3 ML tasks** — regression, classification, association mining
- **5 classifier/regressor** models with head-to-head comparison
- **Per-student risk scores** with feature-level justifications
- **Automated PDF + Excel report** generation with SMTP email dispatch
- **Apriori lift heatmap** identifying co-failing subject combinations

---

## 🧠 Key Learnings

- Building end-to-end ML pipelines with Scikit-learn Pipelines and ColumnTransformer
- Implementing association rule mining for educational pattern discovery
- Designing multi-panel Streamlit apps with dynamic controls
- Generating automated reports (PDF/Excel) and dispatching via SMTP

---

## 👤 Author

**Sutar Sanket Nagnath**  
B.Tech Computer Engineering — MIT Academy of Engineering, Pune (2027)  
GitHub: [@SanketSutar075](https://github.com/SanketSutar075)  
Email: snsutar2004@gmail.com

# app.py  — Final ALL-IN-ONE Student Predictive Analytics Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules

# Optional 3rd-party features
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# PDF optional
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# kaleido optional (Plotly image engine)
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# ---------- compatibility helper for OneHotEncoder ----------
def make_onehot():
    # sklearn historically used 'sparse' param, newer uses 'sparse_output'
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# ---------- helpers ----------
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def df_to_excel_bytes(df_in):
    buf = io.BytesIO()
    df_in.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf

def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    def find(*keys):
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None
    mapping = {
        "student_name": find("student_name","student name","name","fullname"),
        "attendance_rate": find("attendance_rate","attendance %","attendance","present"),
        "credits": find("credits","credit","credit_hours","total_credits"),
        "student_age": find("student_age","age","dob","date_of_birth"),
        "department": find("department","dept"),
        "semester": find("semester","sem"),
        "gpa": find("gpa","final_gpa","grade_point","grade","cgpa"),
        "past_grade_avg": find("past_grade_avg","past_grade","avg_grade","previous_grade"),
        "courses_dropped": find("courses_dropped","courses_drop","dropped"),
        "semester_load": find("semester_load","course_load","load"),
        "dropout": find("dropout","dropout_risk","drop_out","dropped_out")
    }
    return mapping

def build_preprocessor(X):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # use make_onehot for compatibility
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())])
    preprocessor = ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")
    return preprocessor, num_cols, cat_cols

# ---------- Page config ----------
st.set_page_config(layout="wide", page_title="Student Predictive Analytics")
st.title("🎓 Student Predictive Analytics — Professional Dashboard")

# ---------- Themes ----------
THEMES = {
    "1 Modern Blue Gradient": {"bg":"linear-gradient(135deg,#0f1724,#0f3771)","accent":"#00d4ff","card_bg":"rgba(255,255,255,0.04)","text":"#e6f7ff"},
    "2 Dark Premium": {"bg":"linear-gradient(135deg,#0b0f1a,#14121f)","accent":"#f6c85f","card_bg":"rgba(255,255,255,0.02)","text":"#fff6e6"},
    "3 Neon Tech": {"bg":"linear-gradient(135deg,#020024,#090979,#00d4ff)","accent":"#00ffcc","card_bg":"rgba(0,0,0,0.45)","text":"#eaffff"},
    "4 Soft Pastel": {"bg":"linear-gradient(135deg,#fff5f7,#e6f0ff)","accent":"#ff7ab6","card_bg":"rgba(255,255,255,0.85)","text":"#102a43"},
    "5 University Classic": {"bg":"linear-gradient(135deg,#0b3d91,#2b6cb0)","accent":"#ffd166","card_bg":"rgba(255,255,255,0.06)","text":"#fffefc"}
}
st.sidebar.markdown("## Theme")
theme_choice = st.sidebar.selectbox("Choose theme", list(THEMES.keys()), index=0)
theme = THEMES[theme_choice]
st.markdown(f"""
<style>
:root {{
  --accent: {theme['accent']}; 
  --card-bg: {theme['card_bg']};
  --text-clr: {theme['text']};
}}
[data-testid="stAppViewContainer"] {{
  background: {theme['bg']};
  color: var(--text-clr);
}}
.card {{
  background: var(--card-bg);
  padding: 14px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  color: var(--text-clr);
}}
h1,h2,h3,h4 {{ color: var(--text-clr) }}
.stButton>button {{ background: var(--accent); color: #000; border-radius:6px; padding:8px; }}
</style>
""", unsafe_allow_html=True)

# ---------- Upload ----------
st.sidebar.markdown("## Upload dataset")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx","xls"])
if uploaded is None:
    st.info("Upload a student dataset Excel file (GPA/grades, attendance, course pass/fail).")
    st.stop()

df = pd.read_excel(uploaded)
st.markdown('<div class="card"><h4>Dataset preview</h4></div>', unsafe_allow_html=True)
st.dataframe(df.head(6))

mapping = detect_columns(df)
st.write("Detected columns (heuristic):")
st.json(mapping)

# create convenience columns when possible
if mapping["student_name"] is None:
    first = next((c for c in df.columns if "first" in c.lower()), None)
    last = next((c for c in df.columns if "last" in c.lower() or "surname" in c.lower()), None)
    if first and last:
        df["student_name"] = df[first].astype(str).str.strip() + " " + df[last].astype(str).str.strip()
        mapping["student_name"] = "student_name"

if mapping["gpa"] is None:
    gp = next((c for c in df.columns if "grade_point" in c.lower() or "cgpa" in c.lower()), None)
    if gp:
        df["GPA"] = safe_numeric(df[gp])
        mapping["gpa"] = "GPA"

if mapping["attendance_rate"] is None and {"classes_attended","classes_held"}.issubset(df.columns):
    df["attendance_rate"] = (safe_numeric(df["classes_attended"]) / safe_numeric(df["classes_held"])) * 100
    mapping["attendance_rate"] = "attendance_rate"

if mapping["student_age"] is None:
    dob_col = next((c for c in df.columns if "dob" in c.lower() or "date_of_birth" in c.lower()), None)
    if dob_col:
        df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
        df["student_age"] = ((pd.Timestamp.now() - df[dob_col]).dt.days // 365).astype("Int64")
        mapping["student_age"] = "student_age"

# detect apriori candidate columns
apriori_cols = []
for c in df.columns:
    vals = df[c].dropna().unique()
    sval = set([str(x).strip().lower() for x in vals])
    if sval.issubset({"0","1","0.0","1.0","true","false","yes","no"}):
        apriori_cols.append(c)
    elif any(isinstance(x, str) and ("pass" in x.lower() or "fail" in x.lower()) for x in vals):
        apriori_cols.append(c)
if not apriori_cols:
    subj_keys = ['calculus','physics','math','chemistry','biology','linear','algebra','programming','cs','engg','course','subject']
    apriori_cols = [c for c in df.columns if any(k in c.lower() for k in subj_keys) and pd.api.types.is_numeric_dtype(df[c])]

# available tasks
can_regress = all(mapping.get(k) for k in ["attendance_rate","credits","student_age","department","semester","gpa"]) and (mapping.get("gpa") in df.columns and pd.api.types.is_numeric_dtype(df[mapping["gpa"]])) if mapping.get("gpa") else False
can_apriori = len(apriori_cols) >= 1
can_classify = (mapping.get("dropout") in df.columns) and (df[mapping["dropout"]].nunique() <= 2) and all(mapping.get(k) for k in ["attendance_rate","past_grade_avg","courses_dropped","semester_load"])

st.sidebar.markdown("---")
st.sidebar.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
st.sidebar.write("Available tasks:")
st.sidebar.write(f"- Regression: {'✅' if can_regress else '❌'}")
st.sidebar.write(f"- Association Rules: {'✅' if can_apriori else '❌'}")
st.sidebar.write(f"- Classification: {'✅' if can_classify else '❌'}")
if st.sidebar.button("Download cleaned dataset"):
    st.sidebar.download_button("Download cleaned data", data=df_to_excel_bytes(df), file_name="cleaned_student_data.xlsx")

# ---------- Layout controls ----------
left, right = st.columns([1, 2])
with left:
    task = st.selectbox("Select Panel", ["EDA & Charts", "Regression", "Classification", "Association Rules", "Model Comparison", "High-Risk Dashboard", "Reports & Email"])
    st.markdown("---")
    st.markdown("Chart controls")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    x_col = st.selectbox("X (for charts)", options=df.columns.tolist(), index=0)
    y_col = st.selectbox("Y (numeric)", options=numeric_cols, index=0 if numeric_cols else None)
with right:
    st.markdown('<div class="card"><h3>Dashboard</h3></div>', unsafe_allow_html=True)

# ---------- EDA & Charts ----------
if task == "EDA & Charts":
    st.header("Exploratory Data Analysis — Professional Charts")
    st.subheader("Summary statistics")
    st.dataframe(df.describe(include='all').T)

    # missing values
    st.subheader("Missing values (counts)")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing>0])

    # correlation heatmap
    st.subheader("Correlation (interactive heatmap)")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Correlation heatmap shows linear relationships. Use features with high correlation with GPA for regression models.")
    else:
        st.info("Not enough numeric columns to show correlation.")

    # distribution
    if y_col:
        st.subheader(f"Distribution — {y_col}")
        fig = px.histogram(df, x=y_col, nbins=40, marginal="box", title=f"Distribution of {y_col}", color_discrete_sequence=[theme['accent']])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Check skewness / outliers; consider transformations for heavy skew.")

    # box by category
    if x_col in categorical_cols and y_col:
        st.subheader(f"Boxplot — {y_col} by {x_col}")
        fig = px.box(df, x=x_col, y=y_col, points="outliers", title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Boxplots highlight group-wise distributions and outliers to inform stratified modeling.")

    # Attendance vs GPA trend
    if mapping.get("attendance_rate") in df.columns and mapping.get("gpa") in df.columns:
        st.subheader("Attendance Rate vs GPA (Trend & density)")
        fig = px.scatter(df, x=mapping["attendance_rate"], y=mapping["gpa"], trendline="ols", opacity=0.7, title="Attendance vs GPA")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Shows linear trend — higher attendance typically correlates with better GPA; formula from trendline helps estimate effect size.")

    # sunburst dept -> semester -> avg GPA
    if mapping.get("department") in df.columns and mapping.get("semester") in df.columns and mapping.get("gpa") in df.columns:
        st.subheader("Department → Semester → Avg GPA (Sunburst)")
        sun = df.groupby([mapping["department"], mapping["semester"]])[mapping["gpa"]].mean().reset_index(name="avg_gpa")
        fig = px.sunburst(sun, path=[mapping["department"], mapping["semester"]], values="avg_gpa", color="avg_gpa", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Identify departments/semesters with lower performance for targeted curriculum reviews.")

    # radar for normalized features
    st.subheader("Radar — normalized feature comparison")
    radar_feats = st.multiselect("Choose numeric features (3-6)", options=numeric_cols, default=numeric_cols[:4])
    if len(radar_feats) >= 3:
        rad_df = df[radar_feats].dropna()
        norm = (rad_df - rad_df.min()) / (rad_df.max() - rad_df.min())
        means = norm.mean().values
        theta = radar_feats
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=means.tolist()+[means[0]], theta=theta+[theta[0]], fill='toself', name='Average (normalized)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Justification:** Radar helps compare relative magnitudes across features on the same scale.")

# ---------- Regression ----------
elif task == "Regression":
    st.header("Regression — Predict final grade / GPA")
    if not can_regress:
        st.info("Regression not available. Ensure attendance_rate, credits, student_age, department, semester and numeric GPA exist.")
    else:
        default_feats = [mapping["attendance_rate"], mapping["credits"], mapping["student_age"], mapping["department"], mapping["semester"]]
        default_feats = [c for c in default_feats if c in df.columns]
        features = st.multiselect("Select features", options=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])], default=default_feats)
        # pick numeric target
        numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        target = st.selectbox("Select numeric target", options=numeric_cols_all, index=numeric_cols_all.index(mapping["gpa"]) if mapping.get("gpa") in numeric_cols_all else 0)
        model_choice = st.selectbox("Choose model", ["LinearRegression","Ridge","Lasso"])
        test_size = st.slider("Test size %", 10, 50, 20)
        if st.button("Run Regression"):
            X = df[features].copy()
            y = safe_numeric(df[target])
            mask = y.notna()
            X = X[mask]; y = y[mask]
            preproc, num_cols, cat_cols = build_preprocessor(X)
            model = LinearRegression() if model_choice=="LinearRegression" else Ridge() if model_choice=="Ridge" else Lasso()
            pipe = Pipeline([("pre", preproc), ("model", model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = {"MAE": mean_absolute_error(y_test, y_pred), "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))), "R2": r2_score(y_test, y_pred)}
            st.subheader("Performance")
            st.write(f"- MAE: **{metrics['MAE']:.3f}**  |  - RMSE: **{metrics['RMSE']:.3f}**  |  - R²: **{metrics['R2']:.3f}**")
            # plots
            fig = px.scatter(x=y_test, y=y_pred, labels={"x":"Actual","y":"Predicted"}, title="Actual vs Predicted", trendline="ols", color_discrete_sequence=[theme['accent']])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Justification:** Points near diagonal indicate good predictions. R² shows % variance explained; low R² suggests adding features or non-linear models.")
            # residuals
            resid = y_test - y_pred
            fig2 = px.histogram(resid, nbins=40, title="Residuals (Actual - Predicted)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("**Justification:** Residual distribution should be roughly centered at zero; heavy skew or outliers indicate model issues.")
            # feature importance for linear
            try:
                if hasattr(pipe.named_steps["model"], "coef_"):
                    feat_names = num_cols.copy()
                    if cat_cols:
                        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
                        feat_names += list(ohe.get_feature_names_out(cat_cols))
                    coefs = pipe.named_steps["model"].coef_
                    fi = pd.Series(np.abs(coefs), index=feat_names).sort_values(ascending=False).head(20)
                    st.plotly_chart(px.bar(fi, x=fi.values, y=fi.index, orientation="h", title="Top feature importances (abs coef)"), use_container_width=True)
                    st.markdown("**Justification:** Higher absolute coefficient → larger linear effect on target (after scaling).")
            except Exception:
                st.info("Feature importances unavailable for this model.")

# ---------- Classification ----------
elif task == "Classification":
    st.header("Dropout Prediction — Classification")
    if not can_classify:
        st.info("Classification disabled — ensure dropout column and required features exist.")
    else:
        default_feats = [mapping["attendance_rate"], mapping["past_grade_avg"], mapping["courses_dropped"], mapping["semester_load"]]
        default_feats = [c for c in default_feats if c in df.columns]
        feats = st.multiselect("Select features", options=df.columns.tolist(), default=default_feats)
        target_clf = mapping["dropout"]
        st.write(f"Target column: **{target_clf}**")
        clf_choice = st.selectbox("Choose classifier", ["DecisionTree","RandomForest","XGBoost/GradientBoosting"])
        test_size_c = st.slider("Test size %", 10, 50, 20)
        if st.button("Run Classifier"):
            if len(feats) == 0:
                st.error("Select at least one feature.")
            else:
                Xc = df[feats].copy()
                yc = df[target_clf]
                if yc.dtype == object:
                    yc = yc.astype(str).str.lower().map(lambda x: 1 if x in ["1","yes","true","y","t"] else 0)
                yc = safe_numeric(yc).fillna(0).astype(int)
                mask = yc.notna()
                Xc = Xc[mask]; yc = yc[mask]
                preproc_c, num_cols_c, cat_cols_c = build_preprocessor(Xc)
                if clf_choice == "DecisionTree":
                    clf_model = DecisionTreeClassifier(random_state=42)
                elif clf_choice == "RandomForest":
                    clf_model = RandomForestClassifier(n_estimators=150, random_state=42)
                else:
                    if XGBOOST_AVAILABLE:
                        clf_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
                    else:
                        clf_model = GradientBoostingClassifier(random_state=42)
                pipe_c = Pipeline([("pre", preproc_c), ("clf", clf_model)])
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, test_size=test_size_c/100.0, random_state=42, stratify=yc if len(np.unique(yc))>1 else None)
                pipe_c.fit(X_train_c, y_train_c)
                y_pred_c = pipe_c.predict(X_test_c)
                y_proba_c = None
                try:
                    y_proba_c = pipe_c.predict_proba(X_test_c)[:,1]
                except Exception:
                    y_proba_c = None
                metrics_c = {"Accuracy": accuracy_score(y_test_c, y_pred_c), "Precision": precision_score(y_test_c, y_pred_c, zero_division=0), "Recall": recall_score(y_test_c, y_pred_c, zero_division=0), "AUC": roc_auc_score(y_test_c, y_proba_c) if y_proba_c is not None else None}
                st.subheader("Performance")
                st.write(f"- Accuracy: **{metrics_c['Accuracy']:.3f}**  |  - Precision: **{metrics_c['Precision']:.3f}**  |  - Recall: **{metrics_c['Recall']:.3f}**")
                if metrics_c["AUC"] is not None:
                    st.write(f"- AUC: **{metrics_c['AUC']:.3f}**")
                st.markdown("**Justification:** Precision measures correctness among predicted positives; Recall measures catch-rate of actual positives. AUC summarizes ranking performance.")
                # confusion matrix
                cm = confusion_matrix(y_test_c, y_pred_c)
                labels = [str(x) for x in np.unique(y_test_c)]
                fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
                # feature importance
                try:
                    importances = pipe_c.named_steps["clf"].feature_importances_
                    feat_names = num_cols_c.copy()
                    if cat_cols_c:
                        try:
                            ohe = pipe_c.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
                            feat_names += list(ohe.get_feature_names_out(cat_cols_c))
                        except Exception:
                            feat_names += cat_cols_c
                    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
                    st.plotly_chart(px.bar(fi, x=fi.values, y=fi.index, orientation="h", title="Top Feature Importances"), use_container_width=True)
                    st.markdown("**Justification:** Features with higher importance contributed more to predicting dropout risk; use these to inform interventions.")
                except Exception:
                    st.info("Feature importance not available for this classifier.")

                # high-risk students by probability
                st.subheader("High-risk Students (Downloadable)")
                if y_proba_c is None:
                    st.warning("Model does not provide probabilities; using predicted label as risk flag.")
                    probs = y_pred_c
                else:
                    probs = y_proba_c
                pred_df = X_test_c.copy()
                pred_df["true_label"] = y_test_c
                pred_df["pred_label"] = y_pred_c
                pred_df["pred_prob"] = probs
                threshold = st.slider("Risk threshold", 0.1, 0.95, 0.5)
                high_risk = pred_df[pred_df["pred_prob"] >= threshold].sort_values("pred_prob", ascending=False)
                st.dataframe(high_risk.head(200))
                st.download_button("Download high-risk students (xlsx)", data=df_to_excel_bytes(high_risk.reset_index()), file_name="high_risk_students.xlsx")

# ---------- Association Rules ----------
elif task == "Association Rules":
    st.header("Association Rule Mining (Apriori)")
    if not can_apriori:
        st.info("No pass/fail or course columns detected for Apriori.")
    else:
        st.write("Auto-detected candidate columns:", apriori_cols)
        sel_cols = st.multiselect("Select course / pass columns", apriori_cols, default=apriori_cols)
        min_sup = st.slider("Min support", 0.01, 0.5, 0.05)
        min_conf = st.slider("Min confidence", 0.1, 1.0, 0.6)
        if st.button("Run Apriori"):
            passfail = pd.DataFrame()
            for c in sel_cols:
                ser = df[c]
                if pd.api.types.is_numeric_dtype(ser):
                    th = st.number_input(f"Pass threshold for {c}", value=40.0, key=f"th_{c}")
                    passfail[c] = safe_numeric(ser) >= th
                else:
                    passfail[c] = ser.astype(str).str.lower().apply(lambda x: True if "pass" in x else False)
            pf_bool = passfail.fillna(False).astype(bool)
            st.dataframe(pf_bool.head(8))
            freq = apriori(pf_bool, min_support=min_sup, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            if rules.empty:
                st.warning("No rules found")
            else:
                rules = rules.sort_values(["lift","confidence"], ascending=[False, False])
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                st.subheader("Top rules")
                st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].reset_index(drop=True).head(200))
                # lift heatmap for single-item rules
                pairs = rules[(~rules["antecedents"].str.contains(",")) & (~rules["consequents"].str.contains(","))]
                if not pairs.empty:
                    items = list(set(pairs["antecedents"].tolist() + pairs["consequents"].tolist()))
                    lift_mat = pd.DataFrame(0.0, index=items, columns=items)
                    for _, row in pairs.iterrows():
                        lift_mat.loc[row["antecedents"], row["consequents"]] = row["lift"]
                    st.plotly_chart(px.imshow(lift_mat, text_auto=True, title="Lift heatmap (antecedent → consequent)"), use_container_width=True)
                st.markdown("**Justification:** High lift + confidence rules indicate combinations of subject failures (or passes) that co-occur — useful for targeted remedial groupings.")

# ---------- Model Comparison ----------
elif task == "Model Comparison":
    st.header("Multi-model Comparison")
    # Regression leaderboard already available in Regression panel; here we'll show classifier comparisons if possible
    if not can_classify:
        st.info("Classification not available for model comparison.")
    else:
        clf_list = ["DecisionTree","RandomForest","XGBoost"]
        chosen = st.multiselect("Choose classifiers", clf_list, default=["DecisionTree","RandomForest"])
        clf_feats = [mapping["attendance_rate"], mapping["past_grade_avg"], mapping["courses_dropped"], mapping["semester_load"]]
        clf_feats = [c for c in clf_feats if c in df.columns]
        Xc = df[clf_feats].copy()
        yc = df[mapping["dropout"]]
        if yc.dtype == object:
            yc = yc.astype(str).str.lower().map(lambda x: 1 if x in ["1","yes","true","y","t"] else 0)
        yc = safe_numeric(yc).fillna(0).astype(int)
        test_size_c = st.slider("Test size %", 10, 50, 20)
        if st.button("Run classifier comparison"):
            scores = {}
            for m in chosen:
                if m == "DecisionTree":
                    clf = DecisionTreeClassifier(random_state=42)
                elif m == "RandomForest":
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    if XGBOOST_AVAILABLE:
                        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
                    else:
                        clf = GradientBoostingClassifier(random_state=42)
                preproc_c, num_cols_c, cat_cols_c = build_preprocessor(Xc)
                pipe_c = Pipeline([("pre", preproc_c), ("clf", clf)])
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, test_size=test_size_c/100.0, random_state=42, stratify=yc if len(np.unique(yc))>1 else None)
                pipe_c.fit(X_train_c, y_train_c)
                ypred = pipe_c.predict(X_test_c)
                scores[m] = {"Accuracy": accuracy_score(y_test_c, ypred), "Precision": precision_score(y_test_c, ypred, zero_division=0), "Recall": recall_score(y_test_c, ypred, zero_division=0)}
            st.dataframe(pd.DataFrame(scores).T.sort_values("Accuracy", ascending=False))

# ---------- High-Risk Dashboard ----------
elif task == "High-Risk Dashboard":
    st.header("High-Risk Students — Scoring & Justifications")
    if not can_classify:
        st.info("Run classification panel or ensure required columns exist.")
    else:
        features_hr = [mapping["attendance_rate"], mapping["past_grade_avg"], mapping["courses_dropped"], mapping["semester_load"]]
        features_hr = [c for c in features_hr if c in df.columns]
        Xhr = df[features_hr].copy()
        yhr = df[mapping["dropout"]]
        if yhr.dtype == object:
            yhr = yhr.astype(str).str.lower().map(lambda x: 1 if x in ["1","yes","true","y","t"] else 0)
        yhr = safe_numeric(yhr).fillna(0).astype(int)
        # small model to create risk scores
        try:
            preproc_hr, num_hr, cat_hr = build_preprocessor(Xhr)
            clf_hr = RandomForestClassifier(n_estimators=100, random_state=42)
            pipe_hr = Pipeline([("pre", preproc_hr), ("clf", clf_hr)])
            X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(Xhr, yhr, test_size=0.25, random_state=42, stratify=yhr if len(np.unique(yhr))>1 else None)
            pipe_hr.fit(X_train_hr, y_train_hr)
            try:
                probs = pipe_hr.predict_proba(Xhr)[:,1]
            except Exception:
                probs = pipe_hr.predict(Xhr)
            df_hr = Xhr.copy()
            df_hr["student_name"] = df.get(mapping["student_name"], df.index.astype(str))
            df_hr["risk_score"] = probs
            df_hr_sorted = df_hr.sort_values("risk_score", ascending=False)
            st.subheader("Top high-risk students")
            st.dataframe(df_hr_sorted[["student_name"] + features_hr + ["risk_score"]].head(200))
            st.download_button("Download high-risk list (xlsx)", data=df_to_excel_bytes(df_hr_sorted.reset_index()), file_name="high_risk_students.xlsx")
            # short justifications per student (top numeric contributors)
            try:
                importances = pipe_hr.named_steps["clf"].feature_importances_
                feat_names = num_hr.copy()
                if cat_hr:
                    try:
                        ohe = pipe_hr.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
                        feat_names += list(ohe.get_feature_names_out(cat_hr))
                    except Exception:
                        feat_names += cat_hr
                st.subheader("Sample justifications (top contributing features per student)")
                sample = df_hr_sorted.head(50).copy()
                justs = []
                for idx, row in sample.iterrows():
                    reasons = []
                    for f in num_hr:
                        if f in Xhr.columns:
                            col_mean = Xhr[f].mean()
                            col_std = Xhr[f].std() if Xhr[f].std() != 0 else 1.0
                            z = (row[f] - col_mean) / col_std
                            # importance
                            try:
                                imp_idx = feat_names.index(f)
                                imp = importances[imp_idx] if imp_idx < len(importances) else 0
                            except Exception:
                                imp = 0
                            score = z * imp
                            reasons.append((f, score))
                    reasons_sorted = sorted(reasons, key=lambda x: abs(x[1]), reverse=True)[:3]
                    justs.append("; ".join([f"{r[0]}:{r[1]:.2f}" for r in reasons_sorted]))
                sample["justification"] = justs
                st.dataframe(sample[["student_name"] + features_hr + ["risk_score","justification"]].head(50))
            except Exception:
                st.info("Could not compute per-student justifications.")
        except Exception as e:
            st.error(f"Error preparing high-risk dashboard: {e}")

# ---------- Reports & Email ----------
elif task == "Reports & Email":
    st.header("PDF & Excel Reports — Generate / Email")
    # prepare snapshots
    metrics_snapshot = {}
    samples = []
    if can_regress:
        try:
            feats = [mapping["attendance_rate"], mapping["credits"], mapping["student_age"], mapping["department"], mapping["semester"]]
            feats = [c for c in feats if c in df.columns]
            Xr = df[feats].copy()
            yr = safe_numeric(df[mapping["gpa"]])
            mask = yr.notna()
            Xr = Xr[mask]; yr = yr[mask]
            preproc_r, ncols_r, ccols_r = build_preprocessor(Xr)
            model_r = LinearRegression()
            pipe_r = Pipeline([("pre", preproc_r), ("model", model_r)])
            X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
            pipe_r.fit(X_tr, y_tr)
            ypr = pipe_r.predict(X_te)
            metrics_snapshot["Regression"] = {"MAE": mean_absolute_error(y_te, ypr), "RMSE": float(np.sqrt(mean_squared_error(y_te, ypr))), "R2": r2_score(y_te, ypr)}
            sample_reg = X_te.head(20).copy()
            sample_reg["actual"] = y_te.head(20).values
            sample_reg["predicted"] = ypr[:20]
            samples.append(("Regression sample", sample_reg))
        except Exception:
            metrics_snapshot["Regression"] = {"status":"error"}

    if can_classify:
        try:
            fc = [mapping["attendance_rate"], mapping["past_grade_avg"], mapping["courses_dropped"], mapping["semester_load"]]
            fc = [c for c in fc if c in df.columns]
            Xc = df[fc].copy()
            yc = df[mapping["dropout"]]
            if yc.dtype == object:
                yc = yc.astype(str).str.lower().map(lambda x: 1 if x in ["1","yes","true","y","t"] else 0)
            yc = safe_numeric(yc).fillna(0).astype(int)
            preproc_c, ncols_c, ccols_c = build_preprocessor(Xc)
            clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
            pipe2 = Pipeline([("pre", preproc_c), ("clf", clf2)])
            Xt, Xv, yt, yv = train_test_split(Xc, yc, test_size=0.2, random_state=42)
            pipe2.fit(Xt, yt)
            ypv = pipe2.predict(Xv)
            ypv_proba = None
            try:
                ypv_proba = pipe2.predict_proba(Xv)[:,1]
            except Exception:
                pass
            metrics_snapshot["Classification"] = {"Accuracy": accuracy_score(yv, ypv), "Precision": precision_score(yv, ypv, zero_division=0), "Recall": recall_score(yv, ypv, zero_division=0), "AUC": roc_auc_score(yv, ypv_proba) if ypv_proba is not None else None}
            sample_clf = Xv.head(20).copy()
            sample_clf["true"] = yv.head(20).values
            sample_clf["pred"] = ypv[:20]
            samples.append(("Classification sample", sample_clf))
        except Exception:
            metrics_snapshot["Classification"] = {"status":"error"}

    st.subheader("Create PDF report (optional)")
    report_title = st.text_input("PDF report title", value=f"Student Analytics Report — {datetime.now().strftime('%Y-%m-%d')}")
    if st.button("Generate PDF"):
        if not FPDF_AVAILABLE:
            st.error("FPDF not installed. Install `fpdf` to enable PDF export.")
        else:
            charts = []
            try:
                if 'sample_reg' in locals():
                    charts.append(px.scatter(sample_reg, x="actual", y="predicted", title="Sample Actual vs Predicted"))
                if 'sample_clf' in locals():
                    charts.append(px.histogram(sample_clf["pred"], title="Classifier sample predictions"))
            except Exception:
                pass
            try:
                # create pdf bytes
                def create_pdf_bytes(title, metrics_sections, sample_tables=None, chart_figs=None):
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=10)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 8, title, ln=True)
                    pdf.ln(4)
                    pdf.set_font("Arial", size=10)
                    for sec, md in metrics_sections.items():
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 6, sec, ln=True)
                        pdf.set_font("Arial", size=9)
                        for k,v in md.items():
                            pdf.multi_cell(0, 6, f"- {k}: {v}")
                        pdf.ln(2)
                    if sample_tables:
                        for tname, dfr in sample_tables:
                            pdf.set_font("Arial", "B", 11)
                            pdf.cell(0, 6, tname, ln=True)
                            pdf.set_font("Arial", size=8)
                            rows = dfr.head(8)
                            if rows.empty:
                                pdf.multi_cell(0, 6, "(no rows)")
                            else:
                                cols = rows.columns.tolist()
                                pdf.multi_cell(0, 5, " | ".join(cols))
                                for _, r in rows.iterrows():
                                    pdf.multi_cell(0, 5, " | ".join([str(r[c])[:18] for c in cols]))
                            pdf.ln(3)
                    if chart_figs and KALEIDO_AVAILABLE:
                        for fig in chart_figs:
                            try:
                                img = fig.to_image(format="png", engine="kaleido")
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                tmp.write(img); tmp.close()
                                pdf.add_page()
                                pdf.image(tmp.name, w=180)
                                os.unlink(tmp.name)
                            except Exception:
                                pass
                    out = io.BytesIO()
                    pdf.output(out)
                    out.seek(0)
                    return out
                pdf_bytes = create_pdf_bytes(report_title, metrics_snapshot, sample_tables=samples, chart_figs=charts if KALEIDO_AVAILABLE else None)
                st.success("PDF generated")
                st.download_button("Download PDF", data=pdf_bytes, file_name="student_analytics_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF creation failed: {e}")

    # Email sending
    st.subheader("Send report via email (SMTP)")
    smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", value=587)
    smtp_user = st.text_input("SMTP user (sender email)")
    smtp_pass = st.text_input("SMTP password (app password recommended)", type="password")
    email_to = st.text_input("Recipient email(s) (comma-separated)")
    email_subject = st.text_input("Email subject", value="Student Analytics Report")
    email_body = st.text_area("Email body", value="Please find the attached student analytics report and high-risk list.")

    if st.button("Send email with latest PDF + high-risk list"):
        if not (smtp_server and smtp_user and smtp_pass and email_to):
            st.error("Provide SMTP server, username, password and recipient.")
        else:
            # create attachments
            attachments = []
            if FPDF_AVAILABLE:
                try:
                    pdf_bytes = create_pdf_bytes(report_title, metrics_snapshot, sample_tables=samples, chart_figs=None)
                    attachments.append(("student_analytics_report.pdf", pdf_bytes.read(), "application/pdf"))
                except Exception:
                    pass
            try:
                if 'high_risk' in locals() and not high_risk.empty:
                    attachments.append(("high_risk_students.xlsx", df_to_excel_bytes(high_risk.reset_index()).read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
            except Exception:
                pass
            try:
                import smtplib
                from email.message import EmailMessage
                msg = EmailMessage()
                msg["Subject"] = email_subject
                msg["From"] = smtp_user
                msg["To"] = [e.strip() for e in email_to.split(",")]
                msg.set_content(email_body)
                for fname, fbytes, ftype in attachments:
                    maintype, subtype = ftype.split("/")
                    msg.add_attachment(fbytes, maintype=maintype, subtype=subtype, filename=fname)
                with smtplib.SMTP(smtp_server, int(smtp_port)) as smtp:
                    smtp.starttls()
                    smtp.login(smtp_user, smtp_pass)
                    smtp.send_message(msg)
                st.success("Email sent (check spam folder if not in inbox).")
            except Exception as e:
                st.error(f"Email failed: {e}")

# ---------- End ----------
st.markdown("---")
st.write("✅ Full dashboard loaded. Charts are colorful and each chart includes a short justification below it. If you want I can:")
st.write("• Save trained models (.pkl) and add an inference endpoint;")
st.write("• Add authentication;")
st.write("• Produce a PPTX export of the main findings.")

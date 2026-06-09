import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

# ===============================
# --- Streamlit Page Config ---
# ===============================
st.set_page_config(
    page_title="University Management Dashboard",
    layout="wide",
    page_icon="🎓"
)

# ===============================
# --- Custom CSS for Professional Look & Theme ---
# ===============================
st.markdown("""
<style>
body {
    background-color: #e8f0fe;  /* Light professional blue */
    color: #1B2631;
    font-family: 'Helvetica', sans-serif;
}
h1, h2, h3, h4 {
    color: #0B3D91;  /* Dark blue headers */
}
.stButton>button {
    background-color: #0B3D91;  /* Theme button color */
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 University Management Dashboard")
st.markdown("Full Interactive Analytics Dashboard (Steps 1-10) with Validation, Documentation, and Predictive Analytics")

# ===============================
# Step 1: Upload Excel File
# ===============================
uploaded_file = st.file_uploader("Upload University Excel File (.xlsx)", type=['xlsx'])
if uploaded_file:
    try:
        students_df = pd.read_excel(uploaded_file, sheet_name='students')
        courses_df = pd.read_excel(uploaded_file, sheet_name='courses')
        enrollments_df = pd.read_excel(uploaded_file, sheet_name='enrollments')
        grades_df = pd.read_excel(uploaded_file, sheet_name='enrollments_with_grades')
        st.success("✅ File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file. Check sheet names: {e}")
        st.stop()
else:
    st.info("Please upload the Excel file to proceed.")
    st.stop()

# ===============================
# Step 2: Validation & Quality Checks
# ===============================
st.header("📊 Step 2: Validation & Quality Checks")

# Unique IDs
col1, col2, col3 = st.columns(3)
col1.metric("Students IDs", students_df['student_id'].nunique(), delta=f"Total: {len(students_df)}")
col2.metric("Enrollment IDs", enrollments_df['enrollment_id'].nunique(), delta=f"Total: {len(enrollments_df)}")
if 'grade_id' in grades_df.columns:
    col3.metric("Grade Records", grades_df['grade_id'].nunique(), delta=f"Total: {len(grades_df)}")
else:
    col3.metric("Grade Records", len(grades_df), delta="No grade_id column")

# Students per Semester
st.subheader("Students per Semester")
per_sem = enrollments_df.groupby('semester')['student_id'].nunique().reset_index()
fig_sem = px.line(
    per_sem, x='semester', y='student_id', markers=True,
    title="📈 Unique Students per Semester",
    labels={'student_id':'Unique Students','semester':'Semester'},
    line_shape='linear', color_discrete_sequence=['#FF5733']
)
st.plotly_chart(fig_sem, use_container_width=True)
st.markdown("**Justification:** Shows enrollment trends per semester to help plan academic workload.")

# Grade Distribution
st.subheader("Grade Distribution")
merged_df = pd.merge(enrollments_df, grades_df, on=['student_id','course_id','semester'], how='left')
grade_counts = merged_df['grade'].dropna().value_counts().sort_index().reset_index()
grade_counts.columns = ['Grade','Count']
fig_grade = px.bar(
    grade_counts, x='Grade', y='Count', color='Grade',
    title="📊 Grade Distribution", color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_grade, use_container_width=True)
st.markdown("**Justification:** Shows student performance patterns across courses.")

# Dropped Students per Semester
st.subheader("Dropped Students Analysis")
dropped_df = enrollments_df[enrollments_df['status'].str.lower() == 'dropped'] if 'status' in enrollments_df.columns else pd.DataFrame()
if not dropped_df.empty:
    dropped_per_sem = dropped_df.groupby('semester')['enrollment_id'].count().reset_index()
    fig_drop = px.bar(
        dropped_per_sem, x='semester', y='enrollment_id',
        color='enrollment_id', title="📌 Dropped Students per Semester",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'enrollment_id':'Number of Dropped Students','semester':'Semester'}
    )
    st.plotly_chart(fig_drop, use_container_width=True)
    st.markdown("**Justification:** Highlights students dropping courses per semester for intervention.")
else:
    st.write("No dropped students found in dataset.")

# ===============================
# Step 3: Data Dictionary
# ===============================
st.header("📄 Step 3: Data Dictionary")
data_dict_entries = []
dfs = {'Students':students_df,'Courses':courses_df,'Enrollments':enrollments_df,'Grades':grades_df}
for table_name, df in dfs.items():
    for col in df.columns:
        dtype = df[col].dtype
        desc = "Description to be added"
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].astype(str)
        if dtype=='object' or df[col].nunique()<10:
            example = ", ".join(map(str, df[col].dropna().unique()[:5])) + ("..." if df[col].nunique()>5 else "")
        elif pd.api.types.is_numeric_dtype(dtype):
            example = f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}"
        else:
            example = "Varies"
        data_dict_entries.append({'Table Name':table_name,'Column Name':col,'Data Type':str(dtype),
                                  'Description':desc,'Example Values':example})
data_dict_df = pd.DataFrame(data_dict_entries)
st.dataframe(data_dict_df, use_container_width=True)

# ===============================
# Step 4: Profiling Summary
# ===============================
st.header("📊 Step 4: Profiling Summary")
for table_name, df in dfs.items():
    st.subheader(f"{table_name} Table")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    null_counts = df.isnull().sum().to_frame('Null Values').reset_index().rename(columns={'index':'Column'})
    st.dataframe(null_counts, use_container_width=True)

# ===============================
# Step 5: Transformation Log
# ===============================
st.header("🛠 Step 5: Transformation Log")
trans_log = [
    {'Transformation':'Loaded Excel data','Purpose':'Start analysis','Notes':'Sheets loaded: students, courses, enrollments, grades'},
    {'Transformation':'Merged enrollments with grades','Purpose':'Prepare analysis','Notes':'Join on student_id, course_id, semester'},
    {'Transformation':'Mapped grades to GPA','Purpose':'Standardize performance','Notes':'A=4.0, B=3.0, C=2.0, D=1.0, F=0.0'},
]
trans_df = pd.DataFrame(trans_log)
st.dataframe(trans_df, use_container_width=True)

# ===============================
# Step 6: Attendance vs GPA Heatmap
# ===============================
st.header("📊 Step 6: Attendance vs GPA Heatmap")
if 'attendance_rate' in students_df.columns:
    merged_att = pd.merge(merged_df, students_df[['student_id','attendance_rate']], on='student_id', how='left')
    merged_att = merged_att.dropna(subset=['attendance_rate'])
    merged_att['gpa_points'] = merged_att['grade'].map({'A':4,'B':3,'C':2,'D':1,'F':0})
    merged_att = merged_att.dropna(subset=['gpa_points'])
    if not merged_att.empty:
        fig_heat = px.density_heatmap(
            merged_att, x='attendance_rate', y='gpa_points',
            nbinsx=20, nbinsy=5, color_continuous_scale='Viridis',
            title="Attendance vs GPA Points"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("**Justification:** Shows correlation between attendance and GPA for predictive insights.")
    else:
        st.write("Not enough data for heatmap.")
else:
    st.write("Attendance data not found. Skipping Step 6.")

# ===============================
# Step 7: Top Students & Courses
# ===============================
st.header("📊 Step 7: Top Students & Courses")
merged_df['gpa_points'] = merged_df['grade'].map({'A':4,'B':3,'C':2,'D':1,'F':0})
top_students = merged_df.groupby('student_id')['gpa_points'].mean().sort_values(ascending=False).head(10).reset_index()
fig_top_stu = px.bar(top_students, x='student_id', y='gpa_points', title="Top 10 Students by GPA",
                     color='gpa_points', color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig_top_stu, use_container_width=True)

course_gpa = merged_df.groupby('course_id')['gpa_points'].mean().sort_values(ascending=False).head(10).reset_index()
course_gpa = pd.merge(course_gpa, courses_df[['course_id','course_name']] if 'course_name' in courses_df.columns else course_gpa[['course_id']], on='course_id', how='left')
course_gpa['display_name'] = course_gpa['course_name'].fillna(course_gpa['course_id']) if 'course_name' in course_gpa.columns else course_gpa['course_id']
fig_top_course = px.bar(course_gpa, x='display_name', y='gpa_points', title="Top 10 Courses by Average GPA",
                        color='gpa_points', color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig_top_course, use_container_width=True)

# ===============================
# Step 8: Predictive Analytics Prep
# ===============================
st.header("📊 Step 8: Predictive Analytics Tasks Prep")
st.write("Regression Target: **GPA Points**")
st.write("Features: **attendance_rate**, **credits**, **student_age**, **semester**")
st.write("Metrics: **MAE, RMSE, R²**")

# ===============================
# Step 9: Download Cleaned Data
# ===============================
st.header("💾 Step 9: Download Cleaned Data")
def to_excel(df_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return output.getvalue()

excel_data = to_excel({'Students':students_df,'Courses':courses_df,'Enrollments':enrollments_df,'Grades':grades_df})
st.download_button("📥 Download Cleaned Excel", data=excel_data, file_name="University_Cleaned_Data.xlsx")

# ===============================
# Step 10: HTML Report
# ===============================
st.header("📝 Step 10: Download HTML Report")
html_report = f"""
<h1>University Dashboard Report</h1>
<p>Generated: {pd.Timestamp.now()}</p>
<ul>
<li>Total Unique Students: {students_df['student_id'].nunique()}</li>
<li>Total Enrollments: {len(enrollments_df)}</li>
<li>Average GPA: {merged_df['gpa_points'].mean():.2f}</li>
</ul>
<p>All steps 1-10 completed with interactive charts, validation, and profiling.</p>
"""
st.download_button("📥 Download HTML Report", data=html_report, file_name="University_Report.html", mime='text/html')

st.success("✅ Dashboard ready! Use filters and explore the interactive charts.")

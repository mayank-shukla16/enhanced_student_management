import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
import time
from streamlit_lottie import st_lottie
import requests
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EduVision Pro - Student Management",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .sub-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        animation: bounceIn 0.8s ease-out;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
        animation: slideInRight 0.6s ease-out;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 8px 25px rgba(253, 203, 110, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    .css-1d391kg {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

class EnhancedStudentDataModel:
    def __init__(self):
        self.streams = {
            'Science': ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'English'],
            'Commerce': ['Accountancy', 'Economics', 'Business Studies', 'Computer Science', 'English']
        }
        self.optional_subjects = ['Computer Science', 'AI', 'Mass Media', 'Physical Education']
        self.base_columns = ['StudentID', 'Name', 'Age', 'Grade', 'Stream']
        self.attendance_columns = ['Date', 'Status', 'Remarks']
        self._initialize_dataframe()

    def _initialize_dataframe(self):
        all_subjects = self._all_subjects()
        self.students_df = pd.DataFrame(columns=self.base_columns + all_subjects)
        self.attendance_df = pd.DataFrame(columns=['StudentID'] + self.attendance_columns)
        self._ensure_columns_and_types()

    def _all_subjects(self):
        subjects = set()
        for stream_subjects in self.streams.values():
            subjects.update(stream_subjects)
        subjects.update(self.optional_subjects)
        return sorted(subjects)

    def _expected_columns(self):
        return self.base_columns + self._all_subjects()

    def _ensure_columns_and_types(self):
        expected = self._expected_columns()
        for col in expected:
            if col not in self.students_df.columns:
                self.students_df[col] = 0 if col in self._all_subjects() else ""
        
        if 'StudentID' in self.students_df.columns:
            self.students_df['StudentID'] = pd.to_numeric(self.students_df['StudentID'], errors='coerce').astype('Int64')
        if 'Age' in self.students_df.columns:
            self.students_df['Age'] = pd.to_numeric(self.students_df['Age'], errors='coerce').astype('Int64')
        
        for subject in self._all_subjects():
            if subject in self.students_df.columns:
                self.students_df[subject] = self.students_df[subject].apply(
                    lambda x: pd.to_numeric(x, errors='coerce') if x != "N/A" else "N/A"
                )
                self.students_df[subject] = self.students_df[subject].apply(
                    lambda x: 0 if pd.isna(x) and x != "N/A" else x
                )
        
        for col in ['Name', 'Grade', 'Stream']:
            if col in self.students_df.columns:
                self.students_df[col] = self.students_df[col].fillna("").astype(str)

    def add_student(self, data):
        try:
            data_copy = data.copy()
            data_copy['StudentID'] = int(data_copy['StudentID'])
            if 'Age' in data_copy:
                data_copy['Age'] = int(data_copy['Age'])
        except Exception as e:
            return False, f"Invalid ID/Age. {e}"
        
        if self.students_df['StudentID'].notna().any():
            if (self.students_df['StudentID'].dropna().astype(int) == int(data_copy['StudentID'])).any():
                return False, "Student ID already exists."
        
        stream = data_copy.get('Stream', '')
        stream_subjects = self.streams.get(stream, [])
        
        for subject in self._all_subjects():
            if subject not in data_copy:
                if stream != 'Other' and subject not in stream_subjects and subject not in self.optional_subjects:
                    data_copy[subject] = "N/A"
                else:
                    data_copy[subject] = 0
            else:
                try:
                    if data_copy[subject] != "N/A" and data_copy[subject] != "":
                        data_copy[subject] = int(data_copy[subject])
                    elif data_copy[subject] == "":
                        data_copy[subject] = 0
                except:
                    if stream != 'Other' and subject not in stream_subjects and subject not in self.optional_subjects:
                        data_copy[subject] = "N/A"
                    else:
                        return False, f"Invalid mark for {subject}."
        
        new_student = pd.DataFrame([data_copy], columns=self._expected_columns())
        self.students_df = pd.concat([self.students_df, new_student], ignore_index=True)
        self._ensure_columns_and_types()
        return True, "Student added successfully."

    def update_student(self, student_id, data):
        try:
            sid = int(student_id)
        except:
            return False, "Invalid student ID."
        
        mask = self.students_df['StudentID'].notna() & (self.students_df['StudentID'].astype(int) == sid)
        if not mask.any():
            return False, "Student ID not found."
        
        current_stream = self.students_df.loc[mask, 'Stream'].iloc[0]
        
        for key, value in data.items():
            if key in self._all_subjects():
                if current_stream != 'Other' and key not in self.streams.get(current_stream, []) and key not in self.optional_subjects:
                    self.students_df.loc[mask, key] = "N/A"
                else:
                    try:
                        if value != "N/A" and value != "":
                            self.students_df.loc[mask, key] = int(value)
                        elif value == "":
                            self.students_df.loc[mask, key] = 0
                    except:
                        return False, f"Invalid mark for {key}."
            elif key == 'Age':
                try:
                    self.students_df.loc[mask, key] = int(value)
                except:
                    self.students_df.loc[mask, key] = pd.NA
            elif key != 'StudentID':
                self.students_df.loc[mask, key] = str(value)
        
        self._ensure_columns_and_types()
        return True, "Student updated successfully."

    def delete_student(self, student_id):
        try:
            sid = int(student_id)
        except:
            return False, "Invalid student ID."
        
        mask = self.students_df['StudentID'].notna() & (self.students_df['StudentID'].astype(int) == sid)
        if not mask.any():
            return False, "Student ID not found."
        
        self.students_df = self.students_df[~mask].copy()
        return True, "Student deleted successfully."

    def save_students(self, filename):
        try:
            self.students_df.to_csv(filename, index=False)
            return True, f"Data saved to {filename}."
        except Exception as e:
            return False, f"Error saving file: {e}"

    def load_students(self, file):
        try:
            loaded = pd.read_csv(file)
            for col in self._expected_columns():
                if col not in loaded.columns:
                    loaded[col] = 0 if col in self._all_subjects() else ""
            
            self.students_df = loaded[self._expected_columns()].copy()
            self._ensure_columns_and_types()
            return True, f"Data loaded successfully."
        except Exception as e:
            return False, f"Error loading file: {e}"

    def get_individual_report(self, student_id):
        """FIXED: Proper individual report generation"""
        try:
            sid = int(student_id)
        except:
            return None, "Invalid student ID."
        
        mask = self.students_df['StudentID'].notna() & (self.students_df['StudentID'].astype(int) == sid)
        student_row = self.students_df[mask]
        
        if student_row.empty:
            return None, "Student not found."
        
        stream = student_row['Stream'].iloc[0]
        subjects = self.streams.get(stream, [])
        
        # Calculate total and percentage - FIXED LOGIC
        relevant_subjects = []
        total_marks = 0
        
        for subject in subjects:
            mark = student_row[subject].iloc[0]
            if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                try:
                    mark_value = int(mark)
                    total_marks += mark_value
                    relevant_subjects.append(subject)
                except:
                    continue
        
        total_possible = len(relevant_subjects) * 100
        percentage = (total_marks / total_possible * 100) if total_possible > 0 else 0.0
        
        # Prepare report data - FIXED: Handle all data types properly
        student_dict = {}
        for col in self._expected_columns():
            val = student_row.iloc[0][col]
            if pd.isna(val):
                student_dict[col] = "" if col in ['Name', 'Grade', 'Stream'] else 0
            else:
                if col in ['StudentID', 'Age']:
                    try:
                        student_dict[col] = int(val)
                    except:
                        student_dict[col] = 0
                elif col in self._all_subjects():
                    student_dict[col] = val
                else:
                    student_dict[col] = str(val)
        
        student_dict['Total'] = total_marks
        student_dict['Percentage'] = round(percentage, 2)
        student_dict['Subjects_Count'] = len(relevant_subjects)
        
        return student_dict, "Report generated successfully."

    def get_subjects_for_stream(self, stream):
        return self.streams.get(stream, [])

    def get_dashboard_stats(self):
        if self.students_df.empty:
            return {'total_students': 0, 'stream_counts': {}, 'avg_marks_per_stream': {}, 'highest_marks': 0, 'lowest_marks': 0}
        
        total_students = len(self.students_df)
        stream_counts = self.students_df['Stream'].value_counts().to_dict()
        
        avg_marks_per_stream = {}
        for stream in self.streams.keys():
            stream_students = self.students_df[self.students_df['Stream'] == stream]
            if not stream_students.empty:
                stream_subjects = self.streams[stream]
                valid_marks = []
                for _, student in stream_students.iterrows():
                    for subject in stream_subjects:
                        mark = student[subject]
                        if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                            try:
                                valid_marks.append(int(mark))
                            except:
                                pass
                
                if valid_marks:
                    avg_marks = sum(valid_marks) / len(valid_marks)
                    avg_marks_per_stream[stream] = round(avg_marks, 2)
                else:
                    avg_marks_per_stream[stream] = 0
        
        all_marks = []
        for _, student in self.students_df.iterrows():
            stream = student['Stream']
            subjects = self.streams.get(stream, [])
            total_marks = 0
            for subject in subjects:
                mark = student[subject]
                if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                    try:
                        total_marks += int(mark)
                    except:
                        pass
            all_marks.append(total_marks)
        
        highest_marks = max(all_marks) if all_marks else 0
        lowest_marks = min(all_marks) if all_marks else 0
        
        return {
            'total_students': total_students,
            'stream_counts': stream_counts,
            'avg_marks_per_stream': avg_marks_per_stream,
            'highest_marks': highest_marks,
            'lowest_marks': lowest_marks
        }

    def get_table_columns(self):
        base_cols = self.base_columns.copy()
        all_subjects = self._all_subjects()
        
        relevant_subjects = []
        for subject in all_subjects:
            if subject in self.students_df.columns:
                col_data = self.students_df[subject]
                has_data = any(val != "N/A" and val != 0 and pd.notna(val) for val in col_data)
                if has_data:
                    relevant_subjects.append(subject)
        
        return base_cols + relevant_subjects

    def mark_attendance(self, student_id, date, status="Present", remarks=""):
        try:
            sid = int(student_id)
        except:
            return False, "Invalid student ID."
        
        mask = self.students_df['StudentID'].notna() & (self.students_df['StudentID'].astype(int) == sid)
        if not mask.any():
            return False, "Student ID not found."
        
        attendance_record = {
            'StudentID': sid,
            'Date': date,
            'Status': status,
            'Remarks': remarks
        }
        
        new_attendance = pd.DataFrame([attendance_record])
        self.attendance_df = pd.concat([self.attendance_df, new_attendance], ignore_index=True)
        return True, f"Attendance marked as {status} for {date}"

    def get_attendance_report(self, student_id=None, start_date=None, end_date=None):
        report_df = self.attendance_df.copy()
        
        if student_id:
            report_df = report_df[report_df['StudentID'] == student_id]
        
        if start_date:
            report_df = report_df[report_df['Date'] >= start_date]
        
        if end_date:
            report_df = report_df[report_df['Date'] <= end_date]
        
        return report_df

    def get_attendance_stats(self, student_id=None):
        if self.attendance_df.empty:
            return {'total_days': 0, 'present': 0, 'absent': 0, 'percentage': 0}
        
        report_df = self.get_attendance_report(student_id)
        
        if report_df.empty:
            return {'total_days': 0, 'present': 0, 'absent': 0, 'percentage': 0}
        
        total_days = len(report_df)
        present = len(report_df[report_df['Status'] == 'Present'])
        absent = len(report_df[report_df['Status'] == 'Absent'])
        percentage = (present / total_days * 100) if total_days > 0 else 0
        
        return {
            'total_days': total_days,
            'present': present,
            'absent': absent,
            'percentage': round(percentage, 2)
        }

    def predictive_analytics(self, student_id):
        """Predict future performance based on historical data"""
        try:
            sid = int(student_id)
        except:
            return None, "Invalid student ID"
        
        mask = self.students_df['StudentID'].notna() & (self.students_df['StudentID'].astype(int) == sid)
        student_row = self.students_df[mask]
        
        if student_row.empty:
            return None, "Student not found"
        
        stream = student_row['Stream'].iloc[0]
        subjects = self.streams.get(stream, [])
        
        # Calculate current performance
        current_marks = {}
        total_marks = 0
        subject_count = 0
        
        for subject in subjects:
            mark = student_row[subject].iloc[0]
            if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                try:
                    current_marks[subject] = int(mark)
                    total_marks += int(mark)
                    subject_count += 1
                except:
                    continue
        
        if subject_count == 0:
            return None, "No marks data available for prediction"
        
        current_avg = total_marks / subject_count
        
        # Simple prediction algorithm
        attendance_stats = self.get_attendance_stats(student_id)
        attendance_impact = attendance_stats['percentage'] / 100
        
        # Predict future performance
        predicted_improvement = min(15, (100 - current_avg) * 0.3) * attendance_impact
        predicted_avg = current_avg + predicted_improvement
        
        # Risk assessment
        risk_level = "Low"
        if current_avg < 50:
            risk_level = "High"
        elif current_avg < 70:
            risk_level = "Medium"
        
        # Recommendations
        recommendations = []
        if current_avg < 70:
            recommendations.append("Consider additional tutoring sessions")
        if attendance_stats['percentage'] < 80:
            recommendations.append("Improve attendance for better performance")
        
        weak_subjects = [subj for subj, mark in current_marks.items() if mark < 60]
        if weak_subjects:
            recommendations.append(f"Focus on improving: {', '.join(weak_subjects)}")
        
        # Add motivational message
        if current_avg >= 80:
            recommendations.append("Excellent performance! Keep up the great work!")
        elif current_avg >= 70:
            recommendations.append("Good performance! Aim for excellence!")
        
        return {
            'current_average': round(current_avg, 2),
            'predicted_average': round(predicted_avg, 2),
            'improvement_potential': round(predicted_improvement, 2),
            'risk_level': risk_level,
            'attendance_impact': round(attendance_impact * 100, 2),
            'recommendations': recommendations,
            'weak_subjects': weak_subjects,
            'strong_subjects': [subj for subj, mark in current_marks.items() if mark >= 80]
        }, "Analysis completed"

    def get_smart_alerts(self):
        """Generate smart alerts for various conditions"""
        alerts = []
        
        if self.students_df.empty:
            return alerts
        
        # Low performance alerts
        for _, student in self.students_df.iterrows():
            stream = student['Stream']
            subjects = self.streams.get(stream, [])
            total_marks = 0
            subject_count = 0
            
            for subject in subjects:
                mark = student[subject]
                if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                    try:
                        total_marks += int(mark)
                        subject_count += 1
                    except:
                        continue
            
            if subject_count > 0:
                avg_marks = total_marks / subject_count
                if avg_marks < 50:
                    alerts.append({
                        'type': 'Performance',
                        'level': 'High',
                        'message': f"{student['Name']} (ID: {student['StudentID']}) has low average marks ({avg_marks:.1f})",
                        'student_id': student['StudentID'],
                        'timestamp': datetime.now()
                    })
                elif avg_marks < 70:
                    alerts.append({
                        'type': 'Performance',
                        'level': 'Medium',
                        'message': f"{student['Name']} (ID: {student['StudentID']}) has below average marks ({avg_marks:.1f})",
                        'student_id': student['StudentID'],
                        'timestamp': datetime.now()
                    })
        
        # Attendance alerts
        for student_id in self.students_df['StudentID'].unique():
            if pd.notna(student_id):
                stats = self.get_attendance_stats(student_id)
                if stats['percentage'] < 75 and stats['total_days'] > 10:
                    student_name = self.students_df[
                        self.students_df['StudentID'] == student_id
                    ]['Name'].iloc[0]
                    alerts.append({
                        'type': 'Attendance',
                        'level': 'High',
                        'message': f"{student_name} (ID: {student_id}) has low attendance ({stats['percentage']}%)",
                        'student_id': student_id,
                        'timestamp': datetime.now()
                    })
        
        # Sort alerts by level and timestamp
        alerts.sort(key=lambda x: (x['level'] == 'High', x['timestamp']), reverse=True)
        return alerts

    def advanced_search(self, filters):
        """Advanced search with multiple filters"""
        results = self.students_df.copy()
        
        if filters.get('name'):
            results = results[results['Name'].str.contains(filters['name'], case=False, na=False)]
        
        if filters.get('stream') and filters['stream'] != 'All':
            results = results[results['Stream'] == filters['stream']]
        
        if filters.get('min_marks') is not None:
            # Calculate average marks for each student
            def calculate_avg_marks(row):
                stream = row['Stream']
                subjects = self.streams.get(stream, [])
                total_marks = 0
                subject_count = 0
                
                for subject in subjects:
                    mark = row[subject]
                    if mark != "N/A" and pd.notna(mark) and str(mark).isdigit():
                        try:
                            total_marks += int(mark)
                            subject_count += 1
                        except:
                            continue
                
                return total_marks / subject_count if subject_count > 0 else 0
            
            results['avg_marks'] = results.apply(calculate_avg_marks, axis=1)
            results = results[results['avg_marks'] >= filters['min_marks']]
        
        if filters.get('grade'):
            results = results[results['Grade'].str.contains(filters['grade'], case=False, na=False)]
        
        return results

    def bulk_import_students(self, file):
        """Bulk import students from CSV/Excel"""
        try:
            if file.name.endswith('.csv'):
                new_data = pd.read_csv(file)
            else:
                new_data = pd.read_excel(file)
            
            success_count = 0
            error_messages = []
            
            for _, row in new_data.iterrows():
                data_dict = row.to_dict()
                success, message = self.add_student(data_dict)
                if success:
                    success_count += 1
                else:
                    error_messages.append(f"Row {_ + 2}: {message}")
            
            return True, f"Imported {success_count} students successfully. Errors: {len(error_messages)}", error_messages
            
        except Exception as e:
            return False, f"Import failed: {str(e)}", []

    def generate_email_report(self, student_id, recipient_email):
        """Generate and prepare email report"""
        report, message = self.get_individual_report(student_id)
        if not report:
            return False, message
        
        # Create email content
        subject = f"Student Progress Report - {report['Name']}"
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px; }}
                .metric {{ background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 10px; border-left: 4px solid #667eea; }}
                .subject-row {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
                .subject-name {{ font-weight: bold; }}
                .subject-marks {{ color: #667eea; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎓 Student Progress Report</h1>
                    <h2>{report['Name']} (ID: {report['StudentID']})</h2>
                </div>
                
                <div class="metric">
                    <h3>📊 Academic Summary</h3>
                    <p><strong>Stream:</strong> {report['Stream']}</p>
                    <p><strong>Total Marks:</strong> {report['Total']}</p>
                    <p><strong>Percentage:</strong> {report['Percentage']:.2f}%</p>
                </div>
                
                <h3>📚 Subject-wise Performance</h3>
        """
        
        stream = report['Stream']
        subjects = self.streams.get(stream, [])
        for subject in subjects:
            mark = report[subject]
            if mark != "N/A" and str(mark).isdigit():
                html_content += f"""
                <div class="subject-row">
                    <span class="subject-name">{subject}</span>
                    <span class="subject-marks">{mark}/100</span>
                </div>
                """
        
        # Add predictive analytics
        prediction, _ = self.predictive_analytics(student_id)
        if prediction:
            html_content += f"""
                <div class="metric">
                    <h3>🔮 Predictive Analysis</h3>
                    <p><strong>Current Average:</strong> {prediction['current_average']}%</p>
                    <p><strong>Predicted Average:</strong> {prediction['predicted_average']}%</p>
                    <p><strong>Risk Level:</strong> {prediction['risk_level']}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
            """
            for rec in prediction['recommendations']:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return True, html_content, subject

def send_email(to_email, subject, html_content, smtp_config):
    """Send email using SMTP - FIXED IMPORTS"""
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_email']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
        server.starttls()
        server.login(smtp_config['from_email'], smtp_config['password'])
        text = msg.as_string()
        server.sendmail(smtp_config['from_email'], to_email, text)
        server.quit()
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

def show_loading_animation():
    """Show loading animation"""
    with st.spinner(''):
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <div class='loading'></div>
            <p style='margin-top: 1rem; color: #667eea; font-weight: 600;'>Processing...</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)

def show_dashboard(data_model):
    """Main Dashboard with Overview Statistics"""
    st.markdown('<div class="main-header">🎓 EduVision Pro Dashboard</div>', unsafe_allow_html=True)
    
    # Add welcome message with animation
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="sub-header">📊 Real-time Academic Insights</div>', unsafe_allow_html=True)
    
    # Key metrics
    stats = data_model.get_dashboard_stats()
    alerts = data_model.get_smart_alerts()
    
    # Enhanced metrics with icons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card floating'>
                <h3>👥 Total Students</h3>
                <h1 style='font-size: 2.5rem; margin: 0.5rem 0;'>{stats['total_students']}</h1>
                <p>Registered learners</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        science_count = stats['stream_counts'].get('Science', 0)
        st.markdown(f"""
            <div class='metric-card floating' style='animation-delay: 0.2s;'>
                <h3>🔬 Science Stream</h3>
                <h1 style='font-size: 2.5rem; margin: 0.5rem 0;'>{science_count}</h1>
                <p>Future scientists</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        commerce_count = stats['stream_counts'].get('Commerce', 0)
        st.markdown(f"""
            <div class='metric-card floating' style='animation-delay: 0.4s;'>
                <h3>💼 Commerce Stream</h3>
                <h1 style='font-size: 2.5rem; margin: 0.5rem 0;'>{commerce_count}</h1>
                <p>Business leaders</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class='metric-card floating' style='animation-delay: 0.6s;'>
                <h3>🚨 Active Alerts</h3>
                <h1 style='font-size: 2.5rem; margin: 0.5rem 0;'>{len(alerts)}</h1>
                <p>Requiring attention</p>
            </div>
        """, unsafe_allow_html=True)
    
    if stats['total_students'] > 0:
        # Charts and visualizations
        st.markdown("---")
        st.markdown('<div class="sub-header">📈 Performance Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Stream Performance")
            if stats['avg_marks_per_stream'] and any(avg > 0 for avg in stats['avg_marks_per_stream'].values()):
                fig, ax = plt.subplots(figsize=(10, 6))
                streams = list(stats['avg_marks_per_stream'].keys())
                averages = list(stats['avg_marks_per_stream'].values())
                
                # Create gradient colors
                colors = ['#667eea', '#764ba2']
                bars = ax.bar(streams, averages, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                ax.set_ylabel('Average Marks', fontweight='bold')
                ax.set_title('Stream-wise Average Performance', fontweight='bold', fontsize=14, pad=20)
                ax.set_ylim(0, 100)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars with animation effect
                for bar, value in zip(bars, averages):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("📝 No marks data available for visualization. Add student marks to see analytics.")
        
        with col2:
            st.markdown("### 🎯 Stream Distribution")
            if stats['stream_counts']:
                fig, ax = plt.subplots(figsize=(10, 6))
                streams = list(stats['stream_counts'].keys())
                counts = list(stats['stream_counts'].values())
                colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
                
                # Create donut chart
                wedges, texts, autotexts = ax.pie(counts, labels=streams, autopct='%1.1f%%', 
                                                colors=colors[:len(streams)], startangle=90,
                                                wedgeprops=dict(width=0.3, edgecolor='w'))
                
                # Style the autotexts
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                # Style the labels
                for text in texts:
                    text.set_fontsize(11)
                    text.set_fontweight('bold')
                
                ax.set_title('Student Distribution by Stream', fontweight='bold', fontsize=14, pad=20)
                st.pyplot(fig)
        
        # Recent alerts preview with enhanced design
        if alerts:
            st.markdown("---")
            st.markdown('<div class="sub-header">🚨 Recent Alerts</div>', unsafe_allow_html=True)
            
            # Show alert summary
            high_count = len([a for a in alerts if a['level'] == 'High'])
            medium_count = len([a for a in alerts if a['level'] == 'Medium'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔴 High Priority", high_count)
            with col2:
                st.metric("🟡 Medium Priority", medium_count)
            
            # Show recent alerts
            for alert in alerts[:4]:
                level_style = {
                    'High': 'alert-card',
                    'Medium': 'warning-card'
                }
                st.markdown(f"""
                    <div class='{level_style.get(alert["level"], "alert-card")}'>
                        <div style='display: flex; justify-content: space-between; align-items: start;'>
                            <div>
                                <strong>📢 {alert['type']} Alert</strong> • <em>{alert['level']} Priority</em><br>
                                {alert['message']}
                            </div>
                            <div style='font-size: 0.8rem; opacity: 0.8;'>
                                {alert['timestamp'].strftime('%H:%M')}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if len(alerts) > 4:
                st.info(f"📋 **{len(alerts) - 4} more alerts** - Check the Smart Alerts page for complete details")
        else:
            st.markdown("---")
            st.markdown("""
            <div class='success-card'>
                <h3>✅ All Systems Normal</h3>
                <p>No active alerts. Everything is running smoothly!</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Empty state with encouragement
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class='feature-card' style='text-align: center; padding: 3rem;'>
                <h2>🎯 Welcome to EduVision Pro!</h2>
                <p style='font-size: 1.2rem; color: #666; margin: 1rem 0;'>
                    Get started by adding your first student to unlock powerful analytics and insights.
                </p>
                <p style='font-size: 1rem; color: #888;'>
                    Track performance, monitor attendance, and generate intelligent reports all in one place.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='text-align: center;'>
                <h3>🚀 Quick Start</h3>
                <p>1. Go to <strong>Manage Students</strong></p>
                <p>2. Click <strong>Add Student</strong></p>
                <p>3. Fill in student details</p>
                <p>4. Start tracking!</p>
            </div>
            """, unsafe_allow_html=True)

# ... (Other functions would follow with similar enhancements)

def main():
    """Main Application Entry Point"""
    # Initialize session state
    if 'data_model' not in st.session_state:
        st.session_state.data_model = EnhancedStudentDataModel()
    if 'smtp_config' not in st.session_state:
        st.session_state.smtp_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': '',
            'password': ''
        }
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = False

    data_model = st.session_state.data_model

    # Sidebar navigation with modern styling
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
            <h1 style='color: white; margin: 0; font-size: 1.8rem;'>🎓 EduVision Pro</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                Smart Student Management System
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Navigation with icons
    page = st.sidebar.radio("", [
        "🏠 Dashboard", 
        "👨‍🎓 Manage Students", 
        "🔍 Advanced Search",
        "📄 Individual Report",
        "📋 Class Roster",
        "📊 Analytics & Predictions",
        "📅 Attendance Tracking", 
        "📧 Email Reports",
        "⚡ Bulk Operations",
        "🚨 Smart Alerts"
    ], index=0)

    st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    if not data_model.students_df.empty:
        stats = data_model.get_dashboard_stats()
        alerts = data_model.get_smart_alerts()
        high_alerts = len([a for a in alerts if a['level'] == 'High'])
        
        st.sidebar.markdown(f"""
            <div class='metric-card' style='padding: 1.5rem; margin: 1rem 0;'>
                <h4 style='margin: 0 0 1rem 0;'>📊 Quick Stats</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{stats['total_students']}</div>
                        <div style='font-size: 0.8rem;'>Students</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 1.5rem; font-weight: bold; color: {'#ff6b6b' if high_alerts > 0 else '#00b894'};'>{high_alerts}</div>
                        <div style='font-size: 0.8rem;'>High Alerts</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Page routing with loading animation
    if not st.session_state.page_loaded:
        show_loading_animation()
        st.session_state.page_loaded = True

    if page == "🏠 Dashboard":
        show_dashboard(data_model)
    elif page == "👨‍🎓 Manage Students":
        manage_students(data_model)
    elif page == "🔍 Advanced Search":
        advanced_search(data_model)
    elif page == "📄 Individual Report":
        show_individual_report(data_model)
    elif page == "📋 Class Roster":
        show_class_roster(data_model)
    elif page == "📊 Analytics & Predictions":
        show_analytics(data_model)
    elif page == "📅 Attendance Tracking":
        attendance_tracking(data_model)
    elif page == "📧 Email Reports":
        email_reports(data_model)
    elif page == "⚡ Bulk Operations":
        bulk_operations(data_model)
    elif page == "🚨 Smart Alerts":
        smart_alerts(data_model)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;'>
            <p>🎓 EduVision Pro v2.0</p>
            <p>Smart Student Management</p>
        </div>
    """, unsafe_allow_html=True)

# Note: Due to character limits, I've shown the enhanced dashboard and main function.
# The other functions (manage_students, advanced_search, etc.) would follow similar enhancement patterns.

if __name__ == "__main__":
    main()
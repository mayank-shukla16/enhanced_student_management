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
    
    # Ensure all columns exist
    for col in expected:
        if col not in self.students_df.columns:
            self.students_df[col] = 0 if col in self._all_subjects() else ""
    
    # Convert data types
    if 'StudentID' in self.students_df.columns:
        self.students_df['StudentID'] = pd.to_numeric(self.students_df['StudentID'], errors='coerce')
        # Keep as float for now, convert to Int64 later
        self.students_df['StudentID'] = self.students_df['StudentID'].fillna(0).astype('int64')
    
    if 'Age' in self.students_df.columns:
        self.students_df['Age'] = pd.to_numeric(self.students_df['Age'], errors='coerce')
        self.students_df['Age'] = self.students_df['Age'].fillna(0).astype('int64')
    
    # Handle subject marks
    for subject in self._all_subjects():
        if subject in self.students_df.columns:
            # Convert to numeric, coerce errors to NaN
            self.students_df[subject] = pd.to_numeric(self.students_df[subject], errors='coerce')
            # Fill NaN with appropriate values
            mask = self.students_df[subject].isna()
            if mask.any():
                # Check if subject should be N/A based on stream
                for idx in self.students_df[mask].index:
                    stream = self.students_df.loc[idx, 'Stream']
                    stream_subjects = self.streams.get(stream, [])
                    if stream != 'Other' and subject not in stream_subjects and subject not in self.optional_subjects:
                        self.students_df.loc[idx, subject] = "N/A"
                    else:
                        self.students_df.loc[idx, subject] = 0
    
    # Handle string columns
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
        """Bulk import students from CSV/Excel - FIXED VERSION"""
        try:
            # Check if file is empty
            if file.size == 0:
                return False, "Import failed: Uploaded file is empty", []
            
            # Reset file pointer
            file.seek(0)
            
            # Try to read the file based on extension
            try:
                if file.name.endswith('.csv'):
                    new_data = pd.read_csv(file)
                elif file.name.endswith(('.xlsx', '.xls')):
                    new_data = pd.read_excel(file)
                else:
                    return False, "Import failed: Unsupported file format. Use CSV or Excel files", []
            except Exception as read_error:
                return False, f"Import failed: Cannot read file. Error: {str(read_error)}", []
            
            # Check if data was actually read
            if new_data.empty:
                return False, "Import failed: No data found in file or file is empty", []
            
            # Check for required columns
            required_columns = ['StudentID', 'Name', 'Age', 'Grade', 'Stream']
            missing_columns = [col for col in required_columns if col not in new_data.columns]
            
            if missing_columns:
                return False, f"Import failed: Missing required columns: {', '.join(missing_columns)}", []
            
            success_count = 0
            error_messages = []
            processed_rows = []
            
            # Process each row
            for idx, row in new_data.iterrows():
                try:
                    # Convert row to dictionary
                    data_dict = row.to_dict()
                    
                    # Convert numeric fields
                    if 'StudentID' in data_dict:
                        try:
                            data_dict['StudentID'] = int(float(data_dict['StudentID']))
                        except:
                            error_messages.append(f"Row {idx + 2}: Invalid StudentID format")
                            continue
                    
                    if 'Age' in data_dict:
                        try:
                            if pd.notna(data_dict['Age']):
                                data_dict['Age'] = int(float(data_dict['Age']))
                        except:
                            error_messages.append(f"Row {idx + 2}: Invalid Age format")
                            continue
                    
                    # Handle optional fields
                    for col in new_data.columns:
                        if col not in data_dict or pd.isna(data_dict[col]):
                            # Set default values for missing data
                            if col in self._all_subjects():
                                data_dict[col] = 0
                            elif col in ['Name', 'Grade', 'Stream']:
                                data_dict[col] = ""
                            elif col == 'Age':
                                data_dict[col] = 0
                    
                    # Add student
                    success, message = self.add_student(data_dict)
                    if success:
                        success_count += 1
                        processed_rows.append(data_dict['StudentID'])
                    else:
                        error_messages.append(f"Row {idx + 2}: {message}")
                        
                except Exception as row_error:
                    error_messages.append(f"Row {idx + 2}: Unexpected error - {str(row_error)}")
                    continue
            
            # Summary message
            if success_count > 0:
                message = f"✅ Successfully imported {success_count} out of {len(new_data)} students"
                if error_messages:
                    message += f" ({len(error_messages)} errors)"
                return True, message, error_messages
            else:
                return False, f"❌ Failed to import any students. All {len(new_data)} rows had errors", error_messages
                
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

def manage_students(data_model):
    """Manage Students Page"""
    st.markdown('<div class="main-header">👨‍🎓 Student Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="sub-header">Student Operations</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Operations tabs
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Student", "✏️ Edit Student", "🗑️ Delete Student", "💾 Import/Export"])
    
    with tab1:
        st.markdown("### Add New Student")
        
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                student_id = st.number_input("Student ID", min_value=1, step=1, format="%d")
                name = st.text_input("Full Name")
                age = st.number_input("Age", min_value=10, max_value=25, step=1)
                grade = st.selectbox("Grade", ["9th", "10th", "11th", "12th"])
                stream = st.selectbox("Stream", ["Science", "Commerce", "Other"])
            
            with col2:
                st.markdown("#### Subject Marks (0-100)")
                
                # Show subjects based on stream
                subjects = data_model.get_subjects_for_stream(stream)
                optional_subjects = ["Computer Science", "AI", "Mass Media", "Physical Education"]
                
                marks = {}
                for subject in subjects:
                    marks[subject] = st.slider(f"{subject}", 0, 100, 0)
                
                st.markdown("#### Optional Subjects")
                for subject in optional_subjects:
                    if st.checkbox(f"Add {subject}"):
                        marks[subject] = st.slider(f"{subject} Mark", 0, 100, 0, key=f"opt_{subject}")
            
            if st.form_submit_button("🚀 Add Student", use_container_width=True):
                student_data = {
                    'StudentID': student_id,
                    'Name': name,
                    'Age': age,
                    'Grade': grade,
                    'Stream': stream
                }
                
                # Add subject marks
                for subject, mark in marks.items():
                    student_data[subject] = mark
                
                success, message = data_model.add_student(student_data)
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
    
    with tab2:
        st.markdown("### Edit Existing Student")
        
        if not data_model.students_df.empty:
            student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
            selected_id = st.selectbox("Select Student ID", student_ids)
            
            if selected_id:
                # Get current student data
                report, _ = data_model.get_individual_report(selected_id)
                if report:
                    with st.form("edit_student_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            name = st.text_input("Full Name", value=report['Name'])
                            age = st.number_input("Age", min_value=10, max_value=25, 
                                                value=int(report['Age']), step=1)
                            grade = st.selectbox("Grade", ["9th", "10th", "11th", "12th"], 
                                               index=["9th", "10th", "11th", "12th"].index(report['Grade']) if report['Grade'] in ["9th", "10th", "11th", "12th"] else 0)
                            stream = st.selectbox("Stream", ["Science", "Commerce", "Other"], 
                                                index=["Science", "Commerce", "Other"].index(report['Stream']) if report['Stream'] in ["Science", "Commerce", "Other"] else 0)
                        
                        with col2:
                            st.markdown("#### Update Subject Marks")
                            
                            subjects = data_model.get_subjects_for_stream(stream)
                            for subject in subjects:
                                current_mark = report.get(subject, 0)
                                if current_mark == "N/A":
                                    current_mark = 0
                                marks = st.slider(f"{subject}", 0, 100, int(current_mark))
                                report[subject] = marks
                        
                        if st.form_submit_button("💾 Update Student", use_container_width=True):
                            update_data = {
                                'Name': name,
                                'Age': age,
                                'Grade': grade,
                                'Stream': stream
                            }
                            
                            # Add subject marks
                            for subject in subjects:
                                update_data[subject] = report.get(subject, 0)
                            
                            success, message = data_model.update_student(selected_id, update_data)
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
        else:
            st.info("📝 No students found. Add a student first.")
    
    with tab3:
        st.markdown("### Delete Student")
        
        if not data_model.students_df.empty:
            student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
            selected_id = st.selectbox("Select Student ID to Delete", student_ids, key="delete_select")
            
            if selected_id:
                # Show student info before deletion
                student_info = data_model.students_df[
                    data_model.students_df['StudentID'] == selected_id
                ].iloc[0]
                
                st.warning(f"⚠️ You are about to delete:")
                st.write(f"**Name:** {student_info['Name']}")
                st.write(f"**ID:** {selected_id}")
                st.write(f"**Stream:** {student_info['Stream']}")
                
                if st.button("🗑️ Confirm Delete", type="secondary", use_container_width=True):
                    success, message = data_model.delete_student(selected_id)
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        else:
            st.info("📝 No students found.")
    
    with tab4:
        col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Import Data")
        
        # Add import template download
        st.markdown("**Download Import Template:**")
        template_data = pd.DataFrame(columns=['StudentID', 'Name', 'Age', 'Grade', 'Stream'] + data_model._all_subjects())
        template_csv = template_data.to_csv(index=False)
        
        st.download_button(
            label="📋 Download Template CSV",
            data=template_csv,
            file_name="student_import_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", 
                                       type=['csv', 'xlsx', 'xls'],
                                       help="File should contain columns: StudentID, Name, Age, Grade, Stream")
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / 1024  # Convert to KB
            st.info(f"📄 File: {uploaded_file.name} ({file_size:.1f} KB)")
            
            if st.button("📋 Import Students", use_container_width=True):
                with st.spinner("Importing data..."):
                    success, message, errors = data_model.bulk_import_students(uploaded_file)
                    
                    if success:
                        st.success(message)
                        if errors:
                            with st.expander("⚠️ View Import Errors"):
                                for error in errors:
                                    st.error(error)
                    else:
                        st.error(message)
                        if errors:
                            with st.expander("📋 View Detailed Errors"):
                                for error in errors:
                                    st.error(error)
        
        with col2:
            st.markdown("### 📤 Export Data")
            
            export_format = st.radio("Export Format", ["CSV", "Excel"])
            
            if st.button("💾 Export All Data", use_container_width=True):
                if not data_model.students_df.empty:
                    if export_format == "CSV":
                        csv = data_model.students_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"students_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            data_model.students_df.to_excel(writer, index=False, sheet_name='Students')
                        output.seek(0)
                        st.download_button(
                            label="📥 Download Excel",
                            data=output,
                            file_name=f"students_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                else:
                    st.warning("No data to export")


def advanced_search(data_model):
    """Advanced Search Page"""
    st.markdown('<div class="main-header">🔍 Advanced Search</div>', unsafe_allow_html=True)
    
    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name_filter = st.text_input("Search by Name")
            stream_filter = st.selectbox("Stream", ["All", "Science", "Commerce", "Other"])
        
        with col2:
            grade_filter = st.text_input("Grade (e.g., 11th)")
            min_marks_filter = st.slider("Minimum Average Marks", 0, 100, 0)
        
        with col3:
            sort_by = st.selectbox("Sort By", ["Name", "Stream", "Grade", "Average Marks"])
            sort_order = st.radio("Order", ["Ascending", "Descending"])
        
        if st.form_submit_button("🔍 Search", use_container_width=True):
            filters = {
                'name': name_filter if name_filter else None,
                'stream': stream_filter if stream_filter != "All" else None,
                'grade': grade_filter if grade_filter else None,
                'min_marks': min_marks_filter if min_marks_filter > 0 else None
            }
            
            results = data_model.advanced_search(filters)
            
            if not results.empty:
                # Calculate average marks for each student
                def calculate_avg_marks(row):
                    stream = row['Stream']
                    subjects = data_model.get_subjects_for_stream(stream)
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
                    
                    return round(total_marks / subject_count, 2) if subject_count > 0 else 0
                
                results['Average Marks'] = results.apply(calculate_avg_marks, axis=1)
                
                # Sort results
                ascending = sort_order == "Ascending"
                if sort_by == "Average Marks":
                    results = results.sort_values(by='Average Marks', ascending=ascending)
                else:
                    results = results.sort_values(by=sort_by, ascending=ascending)
                
                # Display results with metrics
                st.success(f"✅ Found {len(results)} students")
                
                # Show key metrics
                if len(results) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_of_avgs = results['Average Marks'].mean()
                        st.metric("📊 Average Performance", f"{avg_of_avgs:.1f}%")
                    
                    with col2:
                        top_student = results.iloc[0]['Name']
                        st.metric("🏆 Top Student", top_student)
                    
                    with col3:
                        stream_dist = results['Stream'].value_counts()
                        st.metric("🎯 Stream Distribution", f"{len(stream_dist)} streams")
                
                # Display table
                display_columns = ['StudentID', 'Name', 'Stream', 'Grade', 'Average Marks']
                st.dataframe(results[display_columns], use_container_width=True)
                
                # Download option
                csv = results[display_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("📝 No students found matching your criteria.")


def show_individual_report(data_model):
    """Individual Report Page"""
    st.markdown('<div class="main-header">📄 Individual Student Report</div>', unsafe_allow_html=True)
    
    if not data_model.students_df.empty:
        student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
        selected_id = st.selectbox("Select Student ID", student_ids)
        
        if selected_id:
            # Get report data
            report, message = data_model.get_individual_report(selected_id)
            prediction, pred_message = data_model.predictive_analytics(selected_id)
            attendance_stats = data_model.get_attendance_stats(selected_id)
            
            if report:
                # Header section
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class='feature-card'>
                        <h2>📋 Student Profile</h2>
                        <p><strong>Name:</strong> {report['Name']}</p>
                        <p><strong>ID:</strong> {report['StudentID']}</p>
                        <p><strong>Age:</strong> {report['Age']}</p>
                        <p><strong>Grade:</strong> {report['Grade']}</p>
                        <p><strong>Stream:</strong> {report['Stream']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>🎯 Overall Score</h3>
                        <h1 style='font-size: 3rem; margin: 1rem 0;'>{report['Percentage']:.1f}%</h1>
                        <p>Total: {report['Total']} / {report['Subjects_Count'] * 100}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3>📅 Attendance</h3>
                        <h1 style='font-size: 3rem; margin: 1rem 0;'>{attendance_stats['percentage']:.1f}%</h1>
                        <p>{attendance_stats['present']} / {attendance_stats['total_days']} days</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Subject-wise performance
                st.markdown("### 📚 Subject-wise Performance")
                
                stream = report['Stream']
                subjects = data_model.get_subjects_for_stream(stream)
                
                cols = st.columns(len(subjects))
                for idx, subject in enumerate(subjects):
                    mark = report.get(subject, 0)
                    if mark == "N/A":
                        continue
                    
                    with cols[idx % len(cols)]:
                        # Color code based on marks
                        color = "#00b894" if mark >= 80 else "#fdcb6e" if mark >= 60 else "#ff6b6b"
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; border-radius: 10px; 
                                    background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                                    border: 2px solid {color}44;'>
                            <h4 style='margin: 0;'>{subject}</h4>
                            <h2 style='margin: 0.5rem 0; color: {color};'>{mark}/100</h2>
                            <div style='height: 10px; background: #eee; border-radius: 5px; overflow: hidden;'>
                                <div style='width: {mark}%; height: 100%; background: {color};'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Predictive Analytics
                if prediction:
                    st.markdown("### 🔮 Predictive Analytics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='feature-card'>
                            <h3>📈 Performance Trend</h3>
                            <p><strong>Current Average:</strong> {prediction['current_average']}%</p>
                            <p><strong>Predicted Average:</strong> {prediction['predicted_average']}%</p>
                            <p><strong>Potential Improvement:</strong> +{prediction['improvement_potential']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_color = {
                            'High': '#ff6b6b',
                            'Medium': '#fdcb6e',
                            'Low': '#00b894'
                        }.get(prediction['risk_level'], '#666')
                        
                        st.markdown(f"""
                        <div class='feature-card'>
                            <h3>⚠️ Risk Assessment</h3>
                            <div style='display: flex; align-items: center; margin: 1rem 0;'>
                                <div style='width: 20px; height: 20px; border-radius: 50%; 
                                            background: {risk_color}; margin-right: 10px;'></div>
                                <h2 style='margin: 0; color: {risk_color};'>{prediction['risk_level']} Risk</h2>
                            </div>
                            <p><strong>Attendance Impact:</strong> {prediction['attendance_impact']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    for rec in prediction['recommendations']:
                        st.markdown(f"✅ {rec}")
                    
                    # Weak and Strong subjects
                    if prediction['weak_subjects']:
                        st.markdown(f"#### 📉 Need Improvement: {', '.join(prediction['weak_subjects'])}")
                    
                    if prediction['strong_subjects']:
                        st.markdown(f"#### 📈 Strong Areas: {', '.join(prediction['strong_subjects'])}")
                
                # Download Report
                st.markdown("---")
                if st.button("📥 Download Complete Report", use_container_width=True):
                    # Create comprehensive report
                    report_data = {
                        'Student Information': {
                            'Name': report['Name'],
                            'ID': report['StudentID'],
                            'Age': report['Age'],
                            'Grade': report['Grade'],
                            'Stream': report['Stream']
                        },
                        'Academic Performance': {
                            'Total Marks': report['Total'],
                            'Percentage': f"{report['Percentage']:.2f}%",
                            'Subjects Count': report['Subjects_Count']
                        },
                        'Subject-wise Marks': {
                            subject: report.get(subject, 'N/A') for subject in subjects
                        },
                        'Attendance': attendance_stats
                    }
                    
                    if prediction:
                        report_data['Predictive Analytics'] = prediction
                    
                    # Convert to DataFrame for download
                    report_list = []
                    for category, data in report_data.items():
                        if isinstance(data, dict):
                            for key, value in data.items():
                                report_list.append({'Category': category, 'Metric': key, 'Value': value})
                    
                    report_df = pd.DataFrame(report_list)
                    csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Detailed Report",
                        data=csv,
                        file_name=f"student_report_{report['StudentID']}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error(f"❌ {message}")
    else:
        st.info("📝 No students found. Add students first to generate reports.")


def show_class_roster(data_model):
    """Class Roster Page"""
    st.markdown('<div class="main-header">📋 Class Roster</div>', unsafe_allow_html=True)
    
    if not data_model.students_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            stream_filter = st.selectbox("Filter by Stream", ["All"] + list(data_model.streams.keys()))
        with col2:
            grade_filter = st.selectbox("Filter by Grade", ["All"] + sorted(data_model.students_df['Grade'].unique()))
        with col3:
            sort_by = st.selectbox("Sort by", ["Name", "StudentID", "Stream", "Grade"])
        
        # Apply filters
        filtered_df = data_model.students_df.copy()
        
        if stream_filter != "All":
            filtered_df = filtered_df[filtered_df['Stream'] == stream_filter]
        
        if grade_filter != "All":
            filtered_df = filtered_df[filtered_df['Grade'] == grade_filter]
        
        # Sort data
        filtered_df = filtered_df.sort_values(by=sort_by)
        
        # Display metrics
        st.success(f"📊 Showing {len(filtered_df)} students")
        
        # Stream distribution
        if not filtered_df.empty:
            stream_counts = filtered_df['Stream'].value_counts()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(filtered_df))
            with col2:
                science_count = stream_counts.get('Science', 0)
                st.metric("Science", science_count)
            with col3:
                commerce_count = stream_counts.get('Commerce', 0)
                st.metric("Commerce", commerce_count)
            with col4:
                avg_age = filtered_df['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
        
        # Display roster with expandable details
        st.markdown("### 👥 Student Roster")
        
        for idx, student in filtered_df.iterrows():
            with st.expander(f"🎓 {student['Name']} - ID: {student['StudentID']} - {student['Stream']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Grade:** {student['Grade']}")
                    st.write(f"**Age:** {student['Age']}")
                    
                    # Get performance summary
                    report, _ = data_model.get_individual_report(student['StudentID'])
                    if report:
                        st.write(f"**Total Marks:** {report.get('Total', 0)}")
                        st.write(f"**Percentage:** {report.get('Percentage', 0):.1f}%")
                
                with col2:
                    # Quick subject marks
                    stream = student['Stream']
                    subjects = data_model.get_subjects_for_stream(stream)[:3]  # Show first 3 subjects
                    
                    for subject in subjects:
                        mark = student.get(subject, 0)
                        if mark != "N/A":
                            st.write(f"**{subject}:** {mark}/100")
        
        # Download options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Full Roster", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Click to Download CSV",
                    data=csv,
                    file_name=f"class_roster_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Generate Summary Report", use_container_width=True):
                summary = {
                    'Total Students': len(filtered_df),
                    'Stream Distribution': stream_counts.to_dict(),
                    'Average Age': round(filtered_df['Age'].mean(), 1),
                    'Grade Distribution': filtered_df['Grade'].value_counts().to_dict()
                }
                
                summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
                csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="Click to Download Summary",
                    data=csv,
                    file_name=f"roster_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("📝 No students found. Add students to view the class roster.")


def show_analytics(data_model):
    """Analytics & Predictions Page"""
    st.markdown('<div class="main-header">📊 Advanced Analytics</div>', unsafe_allow_html=True)
    
    if not data_model.students_df.empty:
        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Trends", "🎯 Comparative Analysis", 
                                          "🔮 Predictive Insights", "📊 Custom Reports"])
        
        with tab1:
            st.markdown("### 📈 Academic Performance Trends")
            
            # Performance distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Performance Distribution")
                
                # Calculate performance levels
                performance_levels = []
                for _, student in data_model.students_df.iterrows():
                    report, _ = data_model.get_individual_report(student['StudentID'])
                    if report:
                        percentage = report.get('Percentage', 0)
                        if percentage >= 80:
                            performance_levels.append('Excellent')
                        elif percentage >= 70:
                            performance_levels.append('Good')
                        elif percentage >= 60:
                            performance_levels.append('Average')
                        elif percentage >= 50:
                            performance_levels.append('Below Average')
                        else:
                            performance_levels.append('Poor')
                
                if performance_levels:
                    performance_counts = pd.Series(performance_levels).value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#00b894', '#00cec9', '#fdcb6e', '#e17055', '#ff6b6b']
                    wedges, texts, autotexts = ax.pie(performance_counts.values, 
                                                    labels=performance_counts.index,
                                                    autopct='%1.1f%%',
                                                    colors=colors[:len(performance_counts)],
                                                    startangle=90)
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.set_title('Student Performance Distribution', fontweight='bold')
                    st.pyplot(fig)
            
            with col2:
                st.markdown("#### Stream-wise Comparison")
                
                stream_performance = {}
                for stream in data_model.streams.keys():
                    stream_students = data_model.students_df[data_model.students_df['Stream'] == stream]
                    percentages = []
                    
                    for _, student in stream_students.iterrows():
                        report, _ = data_model.get_individual_report(student['StudentID'])
                        if report:
                            percentages.append(report.get('Percentage', 0))
                    
                    if percentages:
                        stream_performance[stream] = np.mean(percentages)
                
                if stream_performance:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    streams = list(stream_performance.keys())
                    performances = list(stream_performance.values())
                    
                    bars = ax.bar(streams, performances, color=['#667eea', '#764ba2'], alpha=0.8)
                    ax.set_ylabel('Average Percentage', fontweight='bold')
                    ax.set_title('Stream Performance Comparison', fontweight='bold')
                    ax.set_ylim(0, 100)
                    
                    for bar, value in zip(bars, performances):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
        
        with tab2:
            st.markdown("### 🎯 Comparative Analysis")
            
            # Select students to compare
            student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
            selected_ids = st.multiselect("Select Students to Compare", student_ids, max_selections=4)
            
            if len(selected_ids) >= 2:
                comparison_data = []
                
                for sid in selected_ids:
                    report, _ = data_model.get_individual_report(sid)
                    prediction, _ = data_model.predictive_analytics(sid)
                    
                    if report and prediction:
                        comparison_data.append({
                            'Student': report['Name'],
                            'ID': report['StudentID'],
                            'Stream': report['Stream'],
                            'Current %': report['Percentage'],
                            'Predicted %': prediction['predicted_average'],
                            'Risk Level': prediction['risk_level']
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.dataframe(comp_df, use_container_width=True)
                    
                    # Visual comparison
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    x = np.arange(len(comp_df))
                    width = 0.35
                    
                    current_bars = ax.bar(x - width/2, comp_df['Current %'], width, 
                                         label='Current', color='#667eea')
                    predicted_bars = ax.bar(x + width/2, comp_df['Predicted %'], width, 
                                           label='Predicted', color='#764ba2')
                    
                    ax.set_xlabel('Students')
                    ax.set_ylabel('Percentage')
                    ax.set_title('Current vs Predicted Performance')
                    ax.set_xticks(x)
                    ax.set_xticklabels(comp_df['Student'])
                    ax.legend()
                    
                    st.pyplot(fig)
        
        with tab3:
            st.markdown("### 🔮 Predictive Insights")
            
            # Get all predictions
            all_predictions = []
            for _, student in data_model.students_df.iterrows():
                prediction, _ = data_model.predictive_analytics(student['StudentID'])
                if prediction:
                    all_predictions.append({
                        'Student': student['Name'],
                        'Current': prediction['current_average'],
                        'Predicted': prediction['predicted_average'],
                        'Improvement': prediction['improvement_potential'],
                        'Risk': prediction['risk_level']
                    })
            
            if all_predictions:
                pred_df = pd.DataFrame(all_predictions)
                
                # Show top improvers
                st.markdown("#### 🚀 Top Improvers")
                top_improvers = pred_df.nlargest(5, 'Improvement')
                st.dataframe(top_improvers, use_container_width=True)
                
                # Risk distribution
                st.markdown("#### ⚠️ Risk Distribution")
                risk_counts = pred_df['Risk'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = {'High': '#ff6b6b', 'Medium': '#fdcb6e', 'Low': '#00b894'}
                    risk_colors = [colors.get(r, '#666') for r in risk_counts.index]
                    
                    ax.bar(risk_counts.index, risk_counts.values, color=risk_colors)
                    ax.set_title('Students by Risk Level', fontweight='bold')
                    st.pyplot(fig)
                
                with col2:
                    avg_improvement = pred_df['Improvement'].mean()
                    high_risk_pct = (pred_df['Risk'] == 'High').mean() * 100
                    
                    st.metric("📈 Average Improvement Potential", f"{avg_improvement:.1f}%")
                    st.metric("🔴 High Risk Students", f"{high_risk_pct:.1f}%")
        
        with tab4:
            st.markdown("### 📊 Custom Analytics Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox("Report Type", [
                    "Performance Summary",
                    "Attendance Analysis",
                    "Stream Comparison",
                    "Grade-wise Analysis"
                ])
                
                include_charts = st.checkbox("Include Charts", value=True)
                include_predictions = st.checkbox("Include Predictions", value=True)
            
            with col2:
                format_type = st.radio("Export Format", ["CSV", "Excel", "PDF (Screenshot)"])
                
                if st.button("📥 Generate Custom Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        # Generate report based on selections
                        report_data = data_model.get_dashboard_stats()
                        
                        # Add custom analysis
                        if report_type == "Performance Summary":
                            analysis = "Performance Summary Report"
                        elif report_type == "Attendance Analysis":
                            analysis = "Attendance Analysis Report"
                        elif report_type == "Stream Comparison":
                            analysis = "Stream Comparison Report"
                        else:
                            analysis = "Grade-wise Analysis Report"
                        
                        st.success(f"✅ Report Generated: {analysis}")
                        
                        # Provide download option
                        if format_type == "CSV":
                            summary_df = pd.DataFrame(list(report_data.items()), 
                                                    columns=['Metric', 'Value'])
                            csv = summary_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download CSV Report",
                                data=csv,
                                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
    else:
        st.info("📝 No students found. Add students to view analytics.")


def attendance_tracking(data_model):
    """Attendance Tracking Page"""
    st.markdown('<div class="main-header">📅 Attendance Tracking</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["✅ Mark Attendance", "📊 View Reports", "📈 Analytics"])
    
    with tab1:
        st.markdown("### ✅ Mark Attendance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Select Date", datetime.now())
            status = st.selectbox("Status", ["Present", "Absent", "Late", "Leave"])
            remarks = st.text_area("Remarks (Optional)")
        
        with col2:
            if not data_model.students_df.empty:
                student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
                student_names = data_model.students_df['Name'].tolist()
                
                # Create dictionary for display
                student_options = {f"{name} (ID: {sid})": sid for name, sid in zip(student_names, student_ids)}
                selected_display = st.selectbox("Select Student", list(student_options.keys()))
                
                if selected_display:
                    selected_id = student_options[selected_display]
                    
                    if st.button("✅ Mark Attendance", use_container_width=True):
                        success, message = data_model.mark_attendance(
                            selected_id, date.strftime('%Y-%m-%d'), status, remarks
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
            else:
                st.info("📝 No students found. Add students first.")
    
    with tab2:
        st.markdown("### 📊 Attendance Reports")
        
        if not data_model.attendance_df.empty:
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                student_filter = st.selectbox("Filter by Student", 
                                            ["All"] + data_model.students_df['Name'].tolist())
            
            with col2:
                date_range = st.date_input("Date Range", 
                                         [datetime.now() - timedelta(days=30), datetime.now()])
            
            # Apply filters
            report_df = data_model.get_attendance_report()
            
            if student_filter != "All":
                student_id = data_model.students_df[
                    data_model.students_df['Name'] == student_filter
                ]['StudentID'].iloc[0]
                report_df = report_df[report_df['StudentID'] == student_id]
            
            if len(date_range) == 2:
                report_df = report_df[
                    (report_df['Date'] >= date_range[0].strftime('%Y-%m-%d')) &
                    (report_df['Date'] <= date_range[1].strftime('%Y-%m-%d'))
                ]
            
            # Display report
            if not report_df.empty:
                st.dataframe(report_df, use_container_width=True)
                
                # Statistics
                stats = {
                    'Total Records': len(report_df),
                    'Present': len(report_df[report_df['Status'] == 'Present']),
                    'Absent': len(report_df[report_df['Status'] == 'Absent']),
                    'Attendance Rate': f"{(len(report_df[report_df['Status'] == 'Present']) / len(report_df) * 100):.1f}%"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)
                
                # Download option
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Attendance Report",
                    data=csv,
                    file_name=f"attendance_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("📝 No attendance records found for the selected filters.")
        else:
            st.info("📝 No attendance records found. Mark attendance first.")
    
    with tab3:
        st.markdown("### 📈 Attendance Analytics")
        
        if not data_model.attendance_df.empty:
            # Overall statistics
            all_stats = data_model.get_attendance_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📅 Total Days", all_stats['total_days'])
            with col2:
                st.metric("✅ Present Days", all_stats['present'])
            with col3:
                st.metric("❌ Absent Days", all_stats['absent'])
            with col4:
                st.metric("📊 Attendance Rate", f"{all_stats['percentage']:.1f}%")
            
            # Individual student analysis
            st.markdown("#### 👤 Student-wise Analysis")
            
            student_attendance = []
            for _, student in data_model.students_df.iterrows():
                stats = data_model.get_attendance_stats(student['StudentID'])
                if stats['total_days'] > 0:
                    student_attendance.append({
                        'Name': student['Name'],
                        'ID': student['StudentID'],
                        'Attendance %': stats['percentage'],
                        'Present': stats['present'],
                        'Total': stats['total_days']
                    })
            
            if student_attendance:
                attendance_df = pd.DataFrame(student_attendance)
                
                # Sort by attendance percentage
                attendance_df = attendance_df.sort_values('Attendance %', ascending=False)
                
                # Display top and bottom performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 🏆 Top Attendees")
                    st.dataframe(attendance_df.head(5), use_container_width=True)
                
                with col2:
                    st.markdown("##### ⚠️ Need Improvement")
                    st.dataframe(attendance_df.tail(5), use_container_width=True)
                
                # Attendance trend chart
                st.markdown("#### 📈 Attendance Trends")
                
                # Group by date
                if not data_model.attendance_df.empty:
                    daily_attendance = data_model.attendance_df.groupby('Date')['Status'].apply(
                        lambda x: (x == 'Present').sum() / len(x) * 100
                    ).reset_index()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(daily_attendance['Date'], daily_attendance['Status'], 
                           marker='o', color='#667eea', linewidth=2)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Attendance %')
                    ax.set_title('Daily Attendance Trend', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("📝 No attendance data available for analytics.")


def email_reports(data_model):
    """Email Reports Page"""
    st.markdown('<div class="main-header">📧 Email Reports</div>', unsafe_allow_html=True)
    
    # SMTP Configuration
    with st.expander("⚙️ SMTP Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.smtp_config['smtp_server'] = st.text_input(
                "SMTP Server", 
                value=st.session_state.smtp_config['smtp_server']
            )
            st.session_state.smtp_config['from_email'] = st.text_input(
                "Sender Email", 
                value=st.session_state.smtp_config['from_email']
            )
        
        with col2:
            st.session_state.smtp_config['smtp_port'] = st.number_input(
                "SMTP Port", 
                value=st.session_state.smtp_config['smtp_port'],
                min_value=1,
                max_value=65535
            )
            st.session_state.smtp_config['password'] = st.text_input(
                "Email Password", 
                value=st.session_state.smtp_config['password'],
                type="password"
            )
    
    if not data_model.students_df.empty:
        # Student selection
        student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
        student_names = data_model.students_df['Name'].tolist()
        
        student_options = {f"{name} (ID: {sid})": sid for name, sid in zip(student_names, student_ids)}
        selected_display = st.selectbox("Select Student", list(student_options.keys()))
        
        if selected_display:
            selected_id = student_options[selected_display]
            
            # Recipient email
            recipient_email = st.text_input("Recipient Email Address")
            
            # Customize email
            with st.expander("✏️ Customize Email Content"):
                email_subject = st.text_input("Email Subject", 
                                            value="Student Progress Report - {student_name}")
                additional_notes = st.text_area("Additional Notes", 
                                              value="Dear Parent/Guardian,\n\nPlease find the progress report attached.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👁️ Preview Report", use_container_width=True):
                    report, message = data_model.get_individual_report(selected_id)
                    if report:
                        st.success("✅ Report generated successfully!")
                        
                        # Show preview
                        with st.expander("📋 Report Preview"):
                            st.json(report)
            
            with col2:
                if st.button("📧 Send Email", use_container_width=True, type="primary"):
                    if not recipient_email:
                        st.error("❌ Please enter recipient email address")
                    elif not st.session_state.smtp_config['from_email']:
                        st.error("❌ Please configure sender email")
                    elif not st.session_state.smtp_config['password']:
                        st.error("❌ Please enter email password")
                    else:
                        with st.spinner("Sending email..."):
                            # Generate email content
                            success, html_content, subject = data_model.generate_email_report(
                                selected_id, recipient_email
                            )
                            
                            if success:
                                # Customize subject
                                student_name = data_model.students_df[
                                    data_model.students_df['StudentID'] == selected_id
                                ]['Name'].iloc[0]
                                
                                final_subject = email_subject.replace("{student_name}", student_name)
                                
                                # Send email
                                send_success, send_message = send_email(
                                    recipient_email,
                                    final_subject,
                                    html_content,
                                    st.session_state.smtp_config
                                )
                                
                                if send_success:
                                    st.success(f"✅ {send_message}")
                                    st.balloons()
                                else:
                                    st.error(f"❌ {send_message}")
                            else:
                                st.error(f"❌ Failed to generate report: {html_content}")
            
            with col3:
                if st.button("📨 Test Connection", use_container_width=True):
                    if st.session_state.smtp_config['from_email'] and st.session_state.smtp_config['password']:
                        with st.spinner("Testing connection..."):
                            try:
                                server = smtplib.SMTP(
                                    st.session_state.smtp_config['smtp_server'],
                                    st.session_state.smtp_config['smtp_port']
                                )
                                server.starttls()
                                server.login(
                                    st.session_state.smtp_config['from_email'],
                                    st.session_state.smtp_config['password']
                                )
                                server.quit()
                                st.success("✅ SMTP Connection Successful!")
                            except Exception as e:
                                st.error(f"❌ Connection failed: {str(e)}")
                    else:
                        st.error("❌ Please enter email and password")
    else:
        st.info("📝 No students found. Add students first to send reports.")


def bulk_operations(data_model):
    """Bulk Operations Page"""
    st.markdown('<div class="main-header">⚡ Bulk Operations</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Bulk Import", "🔄 Bulk Update", "📤 Bulk Export"])
    
    with tab1:
        st.markdown("### 📋 Bulk Import Students")
        
        st.info("""
        **Import Format Guidelines:**
        - CSV or Excel files only
        - Required columns: StudentID, Name, Age, Grade, Stream
        - Optional columns: Subject marks (0-100)
        - Use "N/A" for subjects not applicable
        """)
        
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            # Preview data
            try:
                if uploaded_file.name.endswith('.csv'):
                    preview_df = pd.read_csv(uploaded_file)
                else:
                    preview_df = pd.read_excel(uploaded_file)
                
                st.markdown("#### 📋 Data Preview")
                st.dataframe(preview_df.head(10), use_container_width=True)
                st.write(f"**Total Records:** {len(preview_df)}")
                
                # Import options
                col1, col2 = st.columns(2)
                
                with col1:
                    import_mode = st.radio("Import Mode", ["Add New Only", "Update Existing", "Overwrite All"])
                
                with col2:
                    if st.button("🚀 Start Bulk Import", use_container_width=True):
                        with st.spinner("Importing data..."):
                            success, message, errors = data_model.bulk_import_students(uploaded_file)
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.balloons()
                                
                                if errors:
                                    with st.expander("⚠️ Import Errors"):
                                        for error in errors:
                                            st.error(error)
                            else:
                                st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("### 🔄 Bulk Update Operations")
        
        if not data_model.students_df.empty:
            # Bulk update options
            update_type = st.selectbox("Select Update Type", [
                "Update Marks",
                "Update Stream",
                "Update Grade",
                "Add Attendance"
            ])
            
            if update_type == "Update Marks":
                st.markdown("#### Update Marks for Multiple Students")
                
                # Select students
                student_ids = data_model.students_df['StudentID'].dropna().astype(int).tolist()
                selected_ids = st.multiselect("Select Students", student_ids)
                
                if selected_ids:
                    # Select subject and new mark
                    subject = st.selectbox("Select Subject", data_model._all_subjects())
                    new_mark = st.slider("New Mark", 0, 100, 50)
                    
                    if st.button("💾 Apply Bulk Update", use_container_width=True):
                        success_count = 0
                        error_messages = []
                        
                        for sid in selected_ids:
                            success, message = data_model.update_student(sid, {subject: new_mark})
                            if success:
                                success_count += 1
                            else:
                                error_messages.append(f"Student {sid}: {message}")
                        
                        st.success(f"✅ Updated {success_count} students successfully")
                        
                        if error_messages:
                            with st.expander("⚠️ Update Errors"):
                                for error in error_messages:
                                    st.error(error)
            
            elif update_type == "Update Stream":
                st.markdown("#### Bulk Stream Update")
                
                # Filter students
                current_stream = st.selectbox("Current Stream", ["All"] + list(data_model.streams.keys()))
                new_stream = st.selectbox("New Stream", list(data_model.streams.keys()))
                
                if current_stream != "All":
                    filtered_students = data_model.students_df[
                        data_model.students_df['Stream'] == current_stream
                    ]
                    
                    if not filtered_students.empty:
                        st.write(f"**Students to update:** {len(filtered_students)}")
                        
                        if st.button("🔄 Update Streams", use_container_width=True):
                            success_count = 0
                            for _, student in filtered_students.iterrows():
                                success, _ = data_model.update_student(
                                    student['StudentID'], 
                                    {'Stream': new_stream}
                                )
                                if success:
                                    success_count += 1
                            
                            st.success(f"✅ Updated {success_count} students to {new_stream} stream")
    
    with tab3:
        st.markdown("### 📤 Bulk Export Options")
        
        if not data_model.students_df.empty:
            # Export options
            export_type = st.selectbox("Export Type", [
                "All Student Data",
                "Academic Reports",
                "Attendance Records",
                "Custom Selection"
            ])
            
            format_type = st.radio("Format", ["CSV", "Excel", "JSON"])
            
            if export_type == "Custom Selection":
                # Let user select columns
                all_columns = data_model.get_table_columns()
                selected_columns = st.multiselect("Select Columns", all_columns, default=all_columns[:5])
                
                if selected_columns:
                    export_df = data_model.students_df[selected_columns]
                else:
                    export_df = data_model.students_df
            else:
                export_df = data_model.students_df
            
            if st.button("📥 Generate Export", use_container_width=True):
                if format_type == "CSV":
                    data = export_df.to_csv(index=False)
                    filename = f"bulk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    mime_type = "text/csv"
                elif format_type == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        export_df.to_excel(writer, index=False, sheet_name='Students')
                    data = output.getvalue()
                    filename = f"bulk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:  # JSON
                    data = export_df.to_json(orient='records', indent=2)
                    filename = f"bulk_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    mime_type = "application/json"
                
                st.download_button(
                    label=f"📥 Download {format_type}",
                    data=data,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
        else:
            st.info("📝 No data available for export.")


def smart_alerts(data_model):
    """Smart Alerts Page"""
    st.markdown('<div class="main-header">🚨 Smart Alerts System</div>', unsafe_allow_html=True)
    
    # Get all alerts
    alerts = data_model.get_smart_alerts()
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚨 Total Alerts", len(alerts))
    with col2:
        high_alerts = len([a for a in alerts if a['level'] == 'High'])
        st.metric("🔴 High Priority", high_alerts)
    with col3:
        medium_alerts = len([a for a in alerts if a['level'] == 'Medium'])
        st.metric("🟡 Medium Priority", medium_alerts)
    with col4:
        performance_alerts = len([a for a in alerts if a['type'] == 'Performance'])
        st.metric("📊 Performance Alerts", performance_alerts)
    
    if alerts:
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            level_filter = st.multiselect("Filter by Level", ["High", "Medium"], default=["High", "Medium"])
        
        with col2:
            type_filter = st.multiselect("Filter by Type", ["Performance", "Attendance"], default=["Performance", "Attendance"])
        
        # Apply filters
        filtered_alerts = [
            a for a in alerts 
            if a['level'] in level_filter and a['type'] in type_filter
        ]
        
        st.markdown(f"### 📋 Showing {len(filtered_alerts)} Alerts")
        
        # Display alerts
        for alert in filtered_alerts:
            # Determine card style
            if alert['level'] == 'High':
                card_class = 'alert-card'
                icon = '🔴'
            else:
                card_class = 'warning-card'
                icon = '🟡'
            
            # Format timestamp
            timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            st.markdown(f"""
            <div class='{card_class}'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <div style='flex: 1;'>
                        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                            <span style='font-size: 1.2rem; margin-right: 0.5rem;'>{icon}</span>
                            <strong>{alert['type']} Alert</strong>
                            <span style='margin-left: 1rem; padding: 0.2rem 0.5rem; 
                                      background: rgba(255,255,255,0.2); border-radius: 10px;
                                      font-size: 0.8rem;'>
                                {alert['level']} Priority
                            </span>
                        </div>
                        <p style='margin: 0.5rem 0;'>{alert['message']}</p>
                        <div style='display: flex; gap: 1rem; margin-top: 1rem;'>
                            <a href='#individual-report' style='text-decoration: none;'>
                                <button style='background: rgba(255,255,255,0.2); border: none; 
                                        color: white; padding: 0.3rem 0.8rem; border-radius: 5px;
                                        cursor: pointer; font-size: 0.9rem;'>
                                    👁️ View Student
                                </button>
                            </a>
                            <button onclick='markResolved(this)' style='background: rgba(255,255,255,0.2); 
                                    border: none; color: white; padding: 0.3rem 0.8rem; 
                                    border-radius: 5px; cursor: pointer; font-size: 0.9rem;'>
                                ✅ Mark Resolved
                            </button>
                        </div>
                    </div>
                    <div style='font-size: 0.8rem; opacity: 0.8; text-align: right;'>
                        {timestamp}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Alert statistics
        st.markdown("---")
        st.markdown("### 📈 Alert Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alert type distribution
            alert_types = pd.Series([a['type'] for a in alerts]).value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#667eea', '#764ba2']
            ax.pie(alert_types.values, labels=alert_types.index, autopct='%1.1f%%',
                  colors=colors[:len(alert_types)], startangle=90)
            ax.set_title('Alert Type Distribution', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            # Alert level distribution
            alert_levels = pd.Series([a['level'] for a in alerts]).value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#ff6b6b', '#fdcb6e', '#00b894']
            ax.bar(alert_levels.index, alert_levels.values, color=colors[:len(alert_levels)])
            ax.set_title('Alert Priority Distribution', fontweight='bold')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        # Export alerts
        st.markdown("---")
        if st.button("📥 Export Alerts Report", use_container_width=True):
            alerts_df = pd.DataFrame(alerts)
            csv = alerts_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Alerts CSV",
                data=csv,
                file_name=f"alerts_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown("""
        <div class='success-card' style='text-align: center; padding: 3rem;'>
            <h2>✅ All Clear!</h2>
            <p style='font-size: 1.2rem;'>No active alerts at the moment.</p>
            <p style='font-size: 1rem; opacity: 0.8;'>
                The system is monitoring all students and will alert you if any issues arise.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # JavaScript for marking alerts as resolved
    st.markdown("""
    <script>
    function markResolved(button) {
        button.parentElement.parentElement.parentElement.parentElement.style.opacity = '0.5';
        button.innerText = '✅ Resolved';
        button.disabled = true;
        setTimeout(() => {
            button.parentElement.parentElement.parentElement.parentElement.style.display = 'none';
        }, 1000);
    }
    </script>
    """, unsafe_allow_html=True)

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





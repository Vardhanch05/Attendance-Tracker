import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import re
import numpy as np
import cv2
from difflib import SequenceMatcher

# Set the tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Vardhan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def enhance_image(image):
    # Convert PIL Image to cv2 format
    img_cv = np.array(image)
    
    # Create multiple enhanced versions
    enhanced_images = []
    
    # Basic preprocessing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Version 1: Basic thresholding with lower threshold
    _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    enhanced_images.append(thresh1)
    
    # Version 2: Adaptive thresholding with smaller block size
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    enhanced_images.append(thresh2)
    
    # Version 3: CLAHE enhancement with higher clip limit
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced3 = clahe.apply(gray)
    enhanced_images.append(enhanced3)
    
    # Version 4: Stronger sharpening
    kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced_images.append(sharpened)
    
    # Version 5: Denoising with custom parameters
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    enhanced_images.append(denoised)
    
    return enhanced_images

def extract_text_from_image(image):
    enhanced_versions = enhance_image(image)
    all_text = []
    
    # Tesseract configurations optimized for participant names
    configs = [
        '--oem 3 --psm 6 -c preserve_interword_spaces=1',
        '--oem 3 --psm 11 -c preserve_interword_spaces=1',
        '--oem 3 --psm 3 -c preserve_interword_spaces=1',
        '--oem 3 --psm 4 -c preserve_interword_spaces=1',
        '--oem 3 --psm 1 -c preserve_interword_spaces=1',
        '--oem 3 --psm 7 -c preserve_interword_spaces=1'  # Added single line mode
    ]
    
    # Process each enhanced version
    for img in enhanced_versions:
        for config in configs:
            text = pytesseract.image_to_string(
                img,
                config=config,
                lang='eng'
            )
            all_text.append(text)
    
    # Process original image with additional parameters
    original_cv = np.array(image)
    text = pytesseract.image_to_string(
        original_cv,
        config='--oem 3 --psm 11 -c preserve_interword_spaces=1 --dpi 300',
        lang='eng'
    )
    all_text.append(text)
    
    return '\n'.join(all_text)

def clean_name(name):
    # Remove common Meet UI elements
    ui_elements = [
        'You', 'Present', 'Meeting', 'Chat', 'Participant', 
        'Mute', 'Unmute', 'Camera', 'Microphone', 'Pin',
        'Raise hand', 'Screen', 'Class', 'Host', 'Co-host',
        'Presenter', 'Recording', 'More', 'options', 'Leave',
        'call', 'Share', 'screen', 'Everyone', 'People',
        'Participants', 'Settings', 'Details', 'Layout'
    ]
    
    name = name.strip()
    
    # Remove UI elements while preserving potential name parts
    for element in ui_elements:
        name = re.sub(f'\\b{element}\\b', '', name, flags=re.IGNORECASE)
    
    # Clean special characters while preserving Indian names
    name = re.sub(r'[^\w\s.-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove percentages and numbers
    name = re.sub(r'\d+%', '', name)
    name = re.sub(r'\b\d+\b', '', name)
    
    # Remove common OCR artifacts while preserving connected names
    name = re.sub(r'[_|]', '', name)
    
    return name

def extract_names(text):
    names = set()
    
    # Split text into lines
    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Clean the line
        cleaned_line = clean_name(line)
        
        if len(cleaned_line) <= 2:
            continue
        
        # Pattern for combined names (e.g., VardhanCh)
        combined_pattern = r'\b[A-Z][a-z]+[A-Z][a-z]*\b'
        combined_matches = re.finditer(combined_pattern, cleaned_line)
        for match in combined_matches:
            names.add(match.group())
        
        # Pattern for regular names
        name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',  # First Middle Last
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',                      # First Last
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',                # Multiple capitalized words
            r'\b[A-Z][a-z]+\b',                                    # Single capitalized word
            r'\b[A-Z][a-z]*[A-Z][a-z]+\b'                         # CamelCase names
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, cleaned_line)
            for match in matches:
                names.add(match.group())
        
        # Handle special cases (like VardhanCh)
        words = cleaned_line.split()
        for word in words:
            if (len(word) > 3 and 
                any(c.isupper() for c in word) and 
                not word.isupper() and 
                not any(ui_text.lower() in word.lower() for ui_text in [
                    'you', 'present', 'meeting', 'chat', 'participant',
                    'mute', 'unmute', 'camera', 'microphone', 'pin'
                ])):
                names.add(word)
    
    return list(names)

def calculate_name_similarity(name1, name2):
    # Base similarity using SequenceMatcher
    base_similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    # Handle combined names (e.g., VardhanCh)
    name1_combined = ''.join(name1.split()).lower()
    name2_combined = ''.join(name2.split()).lower()
    combined_similarity = SequenceMatcher(None, name1_combined, name2_combined).ratio()
    
    # Split names into parts
    parts1 = name1.lower().split()
    parts2 = name2.lower().split()
    
    # Compare individual parts
    part_similarities = []
    for p1 in parts1:
        for p2 in parts2:
            if len(p1) > 2 and len(p2) > 2:
                part_similarity = SequenceMatcher(None, p1, p2).ratio()
                part_similarities.append(part_similarity)
    
    # Calculate maximum part similarity
    max_part_similarity = max(part_similarities) if part_similarities else 0
    
    # Combine similarities with weights
    final_similarity = max(
        base_similarity * 0.4 + max_part_similarity * 0.3 + combined_similarity * 0.3,
        combined_similarity
    )
    
    return final_similarity

def mark_attendance(student_name, extracted_names, threshold=0.42):
    # Check for exact match
    if student_name in extracted_names:
        return "Present"
    
    # Check for combined name match
    student_combined = ''.join(student_name.split()).lower()
    for extracted_name in extracted_names:
        extracted_combined = ''.join(extracted_name.split()).lower()
        if student_combined == extracted_combined:
            return "Present"
    
    # Check for similar names
    max_similarity = 0
    
    for extracted_name in extracted_names:
        similarity = calculate_name_similarity(student_name, extracted_name)
        if similarity > max_similarity:
            max_similarity = similarity
    
    if max_similarity >= threshold:
        return "Present"
    
    return "Absent"

def main():
    st.set_page_config(page_title="Meet Attendance Tracker", layout="wide")
    st.title("Google Meet Attendance Tracker")
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        image_file = st.file_uploader("Upload Meet Screenshot", type=['jpg', 'jpeg', 'png'])
    with col2:
        csv_file = st.file_uploader("Upload Student List (CSV)", type=['csv'])
        
    if image_file and csv_file:
        # Process image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Screenshot", use_container_width=True)
        
        with st.spinner("Processing image..."):
            # Extract and clean names
            text = extract_text_from_image(image)
            extracted_names = extract_names(text)
            
            # Process CSV
            df = pd.read_csv(csv_file)
            if 'Name' not in df.columns:
                st.error("CSV must contain a 'Name' column")
                return
            
            # Mark attendance
            results = []
            for student in df['Name']:
                status = mark_attendance(student, extracted_names)
                results.append({
                    'Name': student,
                    'Status': status
                })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Display results
            st.subheader("Attendance Results")
            st.dataframe(results_df, hide_index=True)
            
            # Calculate statistics
            present_count = len([r for r in results if r['Status'] == 'Present'])
            total_count = len(results)
            attendance_percentage = (present_count / total_count) * 100 if total_count > 0 else 0
            
            # Display statistics
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", total_count)
            col2.metric("Present", present_count)
            col3.metric("Attendance Rate", f"{attendance_percentage:.1f}%")
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "attendance.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
import re
import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def clean_text(text):
    # Normalize mathematical symbols
    text = re.sub(r'[\u00F7\x2F]', '/', text)  # Division symbols
    text = re.sub(r'[\u2212\x2D]', '-', text)  # Minus symbols
    text = re.sub(r'[\u2013\u2014]', '-', text)  # Dash symbols
    
    # Handle LaTeX or similar notations
    text = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'\1/\2', text)  # Convert fractions
    text = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', text)  # Convert square roots
    text = re.sub(r'\\left\(|\\right\)', '', text)  # Remove left/right commands
    
    # Whitespace and special characters normalization
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters, preserving mathematical content
    
    # Additional specific corrections can be added here
    
    return text

# Path to your PDF file
pdf_path = 'apug data\MATH136-notes.pdf'  # Update this to the path of your PDF

# Extract text
extracted_text = extract_text_from_pdf(pdf_path)

# Clean text
cleaned_text = clean_text(extracted_text)

# Preview the cleaned text
print(cleaned_text[:10000])  # Print the first 500 characters to check

# Further processing can be done here, such as segmenting into sections
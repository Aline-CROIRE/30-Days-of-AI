# --- 1. Import Libraries ---

import pytesseract
from PIL import Image
import re
import cv2
import os

print("Libraries imported successfully!")

tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not os.path.exists(tesseract_cmd_path):
    print("="*50)
    print("ERROR: Tesseract executable not found at the specified path.")
    print(f"Current path is: '{tesseract_cmd_path}'")
    print("Please update the 'tesseract_cmd_path' variable in this script to your Tesseract installation location.")
    print("="*50)
    exit()

pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
print("Pytesseract path configured successfully.")


# --- 3. Load the Image ---
img_path = 'kl.jpg'
if not os.path.exists(img_path):
    print("="*50)
    print(f"ERROR: Image file '{img_path}' not found.")
    print("Please download a sample receipt image and save it as 'receipt.png' in the same directory.")
    print("="*50)
    exit()

try:
    image = Image.open(img_path)
    print(f"\nImage '{img_path}' loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()


# --- 4. Perform OCR to Extract Raw Text ---
print("\n--- Performing OCR and Extracting Raw Text ---")
# Using lang='eng' for English. Add more languages with '+' e.g., 'eng+fra'
try:
    raw_text = pytesseract.image_to_string(image, lang='eng')
    print("--- RAW EXTRACTED TEXT ---")
    print(raw_text)
    print("--------------------------")
except pytesseract.TesseractNotFoundError:
    print("="*50)
    print("ERROR: Tesseract is not installed or was not found.")
    print("Ensure the 'tesseract_cmd_path' variable is correct and that you have installed Tesseract.")
    print("="*50)
    exit()


# --- 5. Post-Process with Regular Expressions to Find Key Information ---
print("\n--- Post-Processing Text with Regex ---")

# Regex pattern to find lines containing "Total" (case-insensitive) and capture the price
total_pattern = re.compile(r'(?:total|amount).*\s(\$?[\d,]+\.\d{2})', re.IGNORECASE)
# Regex pattern to find a date in various common formats
date_pattern = re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')

total_amount = "Not found"
date = "Not found"

# Search for patterns line by line for better accuracy
for line in raw_text.split('\n'):
    total_match = total_pattern.search(line)
    if total_match:
        total_amount = total_match.group(1)

date_match = date_pattern.search(raw_text)
if date_match:
    date = date_match.group(1)


# --- 6. Output Structured Data ---
print("\n--- Outputting Structured Data ---")
extracted_data = {
    'Invoice Date': date,
    'Total Amount': total_amount
}

print("--- EXTRACTED & STRUCTURED DATA ---")
for key, value in extracted_data.items():
    print(f"{key}: {value}")
print("-----------------------------------")


# --- 7. Display the Original Image for Context ---
print("\nDisplaying the original image...")
# Use OpenCV to display the image in a window
cv2_image = cv2.imread(img_path)
if cv2_image is not None:
    # Resize for better display if the image is too large
    max_height = 800
    height, width = cv2_image.shape[:2]
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        cv2_image_resized = cv2.resize(cv2_image, (new_width, new_height))
    else:
        cv2_image_resized = cv2_image
        
    cv2.imshow("Original Receipt Image (Press any key to close)", cv2_image_resized)
    cv2.waitKey(0) # Wait indefinitely for a key press
    cv2.destroyAllWindows() # Close all OpenCV windows
else:
    print("Could not display image using OpenCV.")
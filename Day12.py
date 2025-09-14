# --- 1. Import Libraries ---
import cv2
from fer import FER
import matplotlib.pyplot as plt
import os

print("Libraries imported successfully!")

# --- 2. Load Image Path ---
img_path = 'people.jpg'

if not os.path.exists(img_path):
    print(f"ERROR: Image file '{img_path}' not found.")
    exit()

img_rgb = plt.imread(img_path) # Use matplotlib to load directly as RGB
print(f"\nImage '{img_path}' loaded successfully.")


# --- 3. Initialize the Advanced Emotion Detector ---
# mtcnn=True uses a powerful deep learning model to find faces, which is more accurate.
print("\n--- Initializing Emotion Detector with MTCNN (this might take a moment)... ---")
emotion_detector = FER(mtcnn=True)


# --- 4. Perform Detection and Analysis in One Step ---
print("\n--- Detecting and Analyzing Faces... ---")
# The FER library will find all faces in the image and analyze them at once.
# This is a more robust approach than using OpenCV's older detector first.
results = emotion_detector.detect_emotions(img_rgb)

if not results:
    print("\nNo faces were detected. Try a different image with clearer, front-facing faces.")
    exit()

print(f"Found and analyzed {len(results)} faces.")
# --- 5. Create the "Emotion Dashboard" Visualization ---
print("\n--- Generating Emotion Dashboard Visualization ---")

num_faces = len(results)
if num_faces == 0:
    print("Could not generate dashboard as no faces were successfully analyzed.")
    exit()

# --- THE FIX: Adjust the figure size to give it more height ---
# We are making the figure taller to create more space between the rows.
# The height is now 6 inches per face instead of 5.
fig, axes = plt.subplots(num_faces, 2, figsize=(10, 6 * num_faces))
# --- END OF FIX ---

if num_faces == 1:
    axes = [axes]

fig.suptitle('Emotion Analysis Dashboard', fontsize=20)

for i, result in enumerate(results):
    (x, y, w, h) = result["box"]
    face_img = img_rgb[y:y+h, x:x+w]
    
    axes[i][0].imshow(face_img)
    axes[i][0].set_title(f'Face #{i+1}')
    axes[i][0].axis('off')

    emotions = result["emotions"]
    emotion_labels = list(emotions.keys())
    emotion_scores = list(emotions.values())
    dominant_emotion = max(emotions, key=emotions.get)
    
    colors = ['#87CEEB' if emo != dominant_emotion else '#FF6347' for emo in emotion_labels]
    
    axes[i][1].barh(emotion_labels, emotion_scores, color=colors)
    axes[i][1].set_title(f'Dominant: {dominant_emotion.capitalize()}')
    axes[i][1].set_xlabel('Confidence Score')
    axes[i][1].set_xlim(0, 1)
    axes[i][1].invert_yaxis()

# tight_layout() is still very important to automatically adjust spacing.
# The rect parameter helps ensure the main title doesn't overlap.
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

print("\nDashboard generation complete!")
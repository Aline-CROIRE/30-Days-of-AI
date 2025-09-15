# --- 1. Import Libraries ---
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
import os
import warnings

# Suppress a common KMeans warning about memory leaks on Windows with MKL
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("Libraries imported successfully!")

# --- 2. Load the Image ---
img_path = r'C:\Users\iTEK\Desktop\30-Days-of-AI\CAT_5098.jpg'

if not os.path.exists(img_path):
    print("="*50)
    print(f"ERROR: Image file not found at '{img_path}'")
    print("Please download a colorful image and save it as 'colorful_image.jpg' in the same directory.")
   
    print("="*50)
    exit()

try:
    # Load the image and normalize pixel values to be between 0 and 1
    img = io.imread(img_path)
    img = img / 255.0
    print(f"\nImage '{img_path}' loaded and normalized successfully.")
except Exception as e:
    print(f"An error occurred while loading the image: {e}")
    exit()


# --- 3. Prepare Data for K-Means ---
# Reshape the image from 3D (height, width, channels) to 2D (pixels, channels)
h, w, c = img.shape
# Ensure we handle grayscale images gracefully if they are loaded
if c > 3: # Handle RGBA images by removing the alpha channel
    img = img[:,:,:3]
    h, w, c = img.shape

X = img.reshape(h * w, c)
print(f"Image reshaped from {img.shape} to {X.shape} for clustering.")


# --- 4. Find the Optimal Number of Clusters (K) with the Elbow Method ---
print("\n--- Finding the optimal K with the Elbow Method (this may take a minute)... ---")
k_values = range(2, 11)
inertia_values = []

for k in k_values:
    print(f"Running K-Means for k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=200)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
print("\nPlease observe the plot to determine the 'elbow' point.")
print("The 'elbow' is the point where the curve's bend is most prominent.")
plt.show()


# --- 5. Perform Final Clustering with the Optimal K ---
# Ask the user to input the optimal K based on the plot
optimal_k = 0
while optimal_k not in k_values:
    try:
        optimal_k_input = input(f"\nEnter the optimal number of clusters (K) from the plot ({min(k_values)}-{max(k_values)}): ")
        optimal_k = int(optimal_k_input)
        if optimal_k not in k_values:
            print("Invalid input. Please enter a number from the range plotted.")
    except ValueError:
        print("Invalid input. Please enter a whole number.")

print(f"\n--- Performing final clustering with K={optimal_k} ---")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto', max_iter=300)
kmeans.fit(X)

# The cluster centers are our dominant colors
dominant_colors = kmeans.cluster_centers_


# --- 6. Create and Display the Color Palette ---
print("\n--- Generating Color Palette ---")
plt.figure(figsize=(8, 2))
plt.imshow([dominant_colors]) # Display the colors as a 1xK image
plt.axis('off')
plt.title(f'Dominant Color Palette (K={optimal_k})')
plt.show()


# --- 7. Generate and Display the Segmented Image ---
print("\n--- Generating Segmented Image ---")
# Get the label (cluster index) for each pixel
labels = kmeans.labels_

# Create the new image by replacing each pixel with its dominant color
segmented_image_data = dominant_colors[labels]

# Reshape the data back to the original image dimensions
segmented_image = segmented_image_data.reshape(h, w, c)

# Display the original and segmented images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(segmented_image)
axes[1].set_title(f'Segmented Image ({optimal_k} Colors)')
axes[1].axis('off')

plt.show()


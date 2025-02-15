from PIL import Image, ImageDraw
import json
import os

# Path to JSON file
json_file_path = r"C:\Users\abdul\OneDrive\Desktop\rice\masking\masked.json"  # Change this to your actual JSON file path

# Load JSON data from file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Image dimensions from JSON
image_width = data["width"]
image_height = data["height"]

# Directory to save masks
output_path = r"C:\Users\abdul\OneDrive\Desktop\Dataset\Basmati\masks"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create a mask for the image (all boxes in one mask)
# Create a blank mask image (black background)
mask = Image.new("L", (image_width, image_height), 0)
draw = ImageDraw.Draw(mask)

# Loop through all boxes and draw them on the mask
for i, box in enumerate(data["boxes"], start=1):
    # Extract bounding box coordinates and correct positioning
    x = int(float(box['x']))
    y = int(float(box['y']))
    width = int(float(box['width']))
    height = int(float(box['height']))

    # Calculate top-left and bottom-right corners
    top_left = (x - width // 2, y - height // 2)
    bottom_right = (x + width // 2, y + height // 2)

    # Draw white rectangle (255) on the mask
    draw.rectangle([top_left, bottom_right], fill=255)

# Save mask image
mask_filename = os.path.join(output_path, "basmati8_mask.jpeg")  # Single mask for the image
mask.save(mask_filename)
print(f"Mask saved: {mask_filename}")

from PIL import Image, ImageDraw
import json
import os

# Path to JSON file
json_file_path = r"C:\Users\abdul\OneDrive\Desktop\rice\masking\masked.json"  # Change this to your actual JSON file path
original_image_path = r"C:\Users\abdul\OneDrive\Desktop\Dataset\Gobindobhog\images\gobindobhog1.jpeg"  # Path to the original image

# Load JSON data from file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Image dimensions from JSON
image_width = data["width"]
image_height = data["height"]

# Load the original image
original_image = Image.open(original_image_path).convert("RGBA")

# Directory to save masks and overlay images
output_path = r"C:\Users\abdul\OneDrive\Desktop\rice\masking\transp_overlay_verify"
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

# Create a semi-transparent mask overlay
mask_overlay = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))  # Transparent background
overlay_draw = ImageDraw.Draw(mask_overlay)

# Loop through all boxes and draw them on the mask overlay (semi-transparent)
for i, box in enumerate(data["boxes"], start=1):
    x = int(float(box['x']))
    y = int(float(box['y']))
    width = int(float(box['width']))
    height = int(float(box['height']))

    # Calculate top-left and bottom-right corners
    top_left = (x - width // 2, y - height // 2)
    bottom_right = (x + width // 2, y + height // 2)
    border_color = (0, 100, 0)  # Dark green
    border_thickness = 3  # Border thickness

    overlay_draw.rectangle([top_left[0] - border_thickness, top_left[1] - border_thickness, 
                        bottom_right[0] + border_thickness, bottom_right[1] + border_thickness], 
                       outline=border_color, width=border_thickness)

    # Draw a semi-transparent white rectangle (255, with transparency)
    overlay_draw.rectangle([top_left, bottom_right], fill=(57, 255, 20, 90)) 

# Combine the original image and the overlay mask
combined_image = Image.alpha_composite(original_image, mask_overlay)

# Save the combined image with overlay
combined_image_filename = os.path.join(output_path, "seeds_with_overlay.png")
combined_image.save(combined_image_filename)
print(f"Image with overlay saved: {combined_image_filename}")

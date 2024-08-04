import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Create a new image with transparent background
image_transparent = Image.new('RGBA', (400, 200), (255, 255, 255, 0))
draw_transparent = ImageDraw.Draw(image_transparent)


# Define fonts and sizes
font_large = ImageFont.truetype("/Users/baoshui/Library/Fonts/DejaVuSans-Bold.ttf", 70)
font_small = ImageFont.truetype("/Users/baoshui/Library/Fonts/DejaVuSans.ttf", 70)

# Define the texts and their sizes
text_large = "LLM"
text_small = "hub"

bbox = draw_transparent.textbbox((0, 0), text_large, font=font_large)
text_large_width = bbox[2] - bbox[0]
text_large_height = bbox[3] - bbox[1]

bbox = draw_transparent.textbbox((0, 0), text_small, font=font_small)
text_small_width = bbox[2] - bbox[0]
text_small_height = bbox[3] - bbox[1]

# Calculate positions
total_width = text_large_width + text_small_width
position_large = ((image_transparent.width - total_width) // 2, (image_transparent.height - text_large_height) // 2)
position_small = (position_large[0] + text_large_width, position_large[1])

# Draw the texts on the image
draw_transparent.text(position_large, text_large, fill="white", font=font_large)
draw_transparent.text(position_small, text_small, fill="grey", font=font_small)

# Display the image
plt.imshow(image_transparent)
plt.axis('off')
plt.show()

# Save the image
image_path_updated = "logo_light.png"
image_transparent.save(image_path_updated)


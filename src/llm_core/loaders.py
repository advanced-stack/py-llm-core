from PIL import Image
import base64
import io


def resize_image(image_path):
    with Image.open(image_path) as img:
        # Get original dimensions
        original_width, original_height = img.size

        # Determine the aspect ratio
        aspect_ratio = original_width / original_height

        # Calculate new dimensions
        if original_width > original_height:
            # Long side is width
            new_width = min(original_width, 2000)
            new_height = int(new_width / aspect_ratio)
            if new_height > 768:
                new_height = 768
                new_width = int(new_height * aspect_ratio)
        else:
            # Long side is height
            new_height = min(original_height, 2000)
            new_width = int(new_height * aspect_ratio)
            if new_width > 768:
                new_width = 768
                new_height = int(new_width * aspect_ratio)

        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the resized image to a BytesIO object
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()


def load_image(image_path):
    resized_image_data = resize_image(image_path)
    return base64.b64encode(resized_image_data).decode("utf-8")

import cv2
import streamlit as st

# Function to perform image resizing
def resize_image(image, width=None, height=None):
    if width is None and height is None:
        return image
    if width is None:
        aspect_ratio = height / float(image.shape[0])
        new_width = int(image.shape[1] * aspect_ratio)
        dim = (new_width, height)
    else:
        aspect_ratio = width / float(image.shape[1])
        new_height = int(image.shape[0] * aspect_ratio)
        dim = (width, new_height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Function to convert image to grayscale
def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to perform image cropping
def crop_image(image, x, y, w, h):
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Function to perform image rotation
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

# Load the image
image_path = './image1.jpg'
original_image = cv2.imread(image_path)

# Perform image resizing
resized_image = resize_image(original_image, width=400)

# Perform grayscale conversion
gray_image = convert_to_grayscale(original_image)

# Perform image cropping
cropped_image = crop_image(original_image, x=100, y=100, w=300, h=300)

# Perform image rotation
rotated_image = rotate_image(original_image, angle=45)

# Display the original and processed images using Streamlit
st.image(original_image, caption='Original Image', use_column_width=True)
st.image(resized_image, caption='Resized Image', use_column_width=True)
st.image(gray_image, caption='Grayscale Image', use_column_width=True)
st.image(cropped_image, caption='Cropped Image', use_column_width=True)
st.image(rotated_image, caption='Rotated Image', use_column_width=True)
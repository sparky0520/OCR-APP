# Import libraries
import cv2
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
from byaldi import RAGMultiModalModel
#from google.colab import files
from IPython.display import display, HTML
import os
import re

# to detect cuda(GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#loading models
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", verbose=0)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

torch.cuda.empty_cache()

#Upload image
# def upload_image():
#     uploaded = files.upload()
#     for filename in uploaded.keys():
#         print(f'Uploaded file: {filename}')
#         return filename

# image_path = upload_image()

# Preprocessing using OpenCV
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at the path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Maintain aspect ratio
    height, width = gray.shape
    if height > width:
        new_height = 1024
        new_width = int((width / height) * new_height)
    else:
        new_width = 1024
        new_height = int((height / width) * new_width)

    resized_image = cv2.resize(gray, (new_width, new_height))

    blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresholded, h=30)
    pil_image = Image.fromarray(denoised)

    return pil_image

 # Call the function and store the result
# pil_image = preprocess_image(image_path)

# display(pil_image) # Now pil_image is accessible here

#extract the text
def extract_text(image_path):
    try:
        processed_image = preprocess_image(image_path)
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "PLease extract the both hindi and english text as they appear in the image"}]}
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[processed_image], padding=True, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=1042)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return extracted_text
    except Exception as e:
        return f"An error occurred during text extraction: {e}"

#keyword searching
def keyword_search(extracted_text, keywords):
    if not keywords:
        return extracted_text, "Please enter a keyword to search and highlight."
    keywords = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]

    highlighted_text = ""

    lines = extracted_text.split('\n')
    for line in lines:
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            line = pattern.sub(lambda m: f'<span style="color: red;">{m.group()}</span>', line)
        highlighted_text += line + '\n'
    return highlighted_text

#OCR and keyword search interface
def ocr_interface(image):
    image_path = "temp_image.png"
    image.save(image_path)
    extracted_text = extract_text(image_path)
    os.remove(image_path)

    return extracted_text, ""
def keyword_interface(extracted_text, keywords):
    highlighted_text = keyword_search(extracted_text, keywords)
    return highlighted_text


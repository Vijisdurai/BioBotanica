import streamlit as st
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from rag_text_generation import generate_medicinal_information  # Import the function for generating medicinal info

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the path to the saved model
MODEL_PATH = r'C:\Users\vijis\OneDrive\Desktop\project\BioBotanica\models\resnet50.pth'

# Class names based on training dataset order (update if needed)
class_names = ["Aloe Vera", "Amla", "Amruta Balli", "Arali", "Hibiscus", "Lemon Grass", "Mint"]

# Load the trained model function
def load_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))  # Load on the right device
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Load the model with the correct number of classes
model = load_model(num_classes=len(class_names))

# Define image preprocessing for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit app title and description
st.title("Medicinal Plant Classifier - BioBotanica")
st.write("Upload an image of a plant, and the model will predict its class and generate information about its medicinal uses.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image for the model
    input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
        predicted_class_idx = predicted.item()
        predicted_class_name = class_names[predicted_class_idx]  # Get the class name

    # Display the predicted plant name
    st.write(f"**Predicted Plant:** {predicted_class_name}")
    st.write("Fetching medicinal information...")

    # Generate medicinal information using RAG and Generative AI
    medicinal_info = generate_medicinal_information(predicted_class_name)

    # Display the generated information
    st.write(f"**Medicinal Uses and Properties:**\n{medicinal_info}")

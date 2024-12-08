import openai
import streamlit as st

# Load OpenAI API key securely from an environment variable
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_medicinal_information(plant_name):
    """
    Uses GPT-4 to generate medicinal information about the specified plant.
    """
    prompt = f'''Provide information about the plant '{plant_name}' in the following format:
- Medicinal Uses: Describe how the plant is used medicinally.
- Traditional Applications: Explain any traditional uses in healthcare or wellness.
- Known Health Benefits: List specific health benefits and the scientific basis, if available.
Summarize this with detail within 1500 tokens.'''

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert botanist and herbalist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=1.0
        )
        # Extract the generated text
        medicinal_info = response['choices'][0]['message']['content'].strip()
        return medicinal_info
    
    except Exception as e:
        return f"An error occurred: {e}"

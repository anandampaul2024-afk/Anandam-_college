
# llmhelper.py
import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyD3f2oZ3m26ATVMupHWRfzxlvmY0ZZ7lG8")

# Choose a stable model
MODEL_NAME = "models/gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)

def generate_explanation(prompt):
    """
    Generate a detailed football analysis for the given prompt.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating explanation: {str(e)}"
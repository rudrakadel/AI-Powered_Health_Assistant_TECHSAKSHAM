import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize NLP components
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined healthcare FAQs and responses
FAQS = {
    "What causes sneezing?": "Sneezing is often caused by irritants like dust, pollen, or allergens. It could also indicate a cold or flu.",
    "What are common flu symptoms?": "Common flu symptoms include fever, cough, body aches, and fatigue.",
    "How can I schedule an appointment?": "You can schedule an appointment by contacting your nearest clinic or using an online booking system.",
    "What should I do if I have a fever?": "Drink plenty of fluids, rest, and monitor your temperature. If it persists, consult a healthcare provider."
}

# Function to preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function for semantic matching
def find_best_match(user_input):
    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)
    faq_embeddings = [semantic_model.encode(faq, convert_to_tensor=True) for faq in FAQS.keys()]
    scores = [util.pytorch_cos_sim(user_embedding, faq_emb).item() for faq_emb in faq_embeddings]
    best_match_idx = scores.index(max(scores))
    return list(FAQS.keys())[best_match_idx], max(scores)

# Healthcare chatbot logic
def healthcare_chatbot(user_input):
    # Preprocess input
    user_input = preprocess_input(user_input).lower()

    # Check for semantic match with FAQs
    best_match, score = find_best_match(user_input)
    if score > 0.7:  # Threshold for semantic similarity
        return FAQS[best_match]

    # Use QA model for general questions
    try:
        response = qa_pipeline(question=user_input, context=" ".join(FAQS.values()))
        return response['answer']
    except Exception as e:
        return "I'm sorry, I couldn't process that. Please try rephrasing your question."

# Main function for Streamlit app
def main():
    st.title("Advanced Healthcare Chatbot")
    st.subheader("Your virtual healthcare assistant")

    user_input = st.text_input("How can I assist you today?", "")
    if st.button("Submit"):
        if user_input.strip():  # Check for non-empty input
            st.write("**User:** ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:** ", response)
        else:
            st.warning("Please enter a query before submitting!")

    # Add FAQ section
    st.write("### Frequently Asked Questions:")
    for question in FAQS.keys():
        if st.button(question):
            st.write("**Answer:** ", FAQS[question])

if __name__ == "__main__":
    main()

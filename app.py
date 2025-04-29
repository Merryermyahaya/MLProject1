from flask import Flask, render_template, request, jsonify
from sentence_transformers.util import cos_sim
import joblib
import torch
import gdown
import os

# Ensure the model file exists
#MODEL_PATH = "chatbot_model.pkl"
#if not os.path.exists(MODEL_PATH):
    #url = "https://drive.google.com/uc?id=1O5A-wC4fMzuW-D207t42TTFPH_EtyoP0"
    #gdown.download(url, MODEL_PATH, quiet=False)

# Load the chatbot object
chatbot = joblib.load(MODEL_PATH)

# Define the chatbot class
class SemanticChatbot:
    def __init__(self, df, index, model):
        self.df = df
        self.index = index
        self.model = model

    def get_response(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        question_embeddings = self.model.encode(self.df["Preprocessed_Question"].tolist(), convert_to_tensor=True)
        scores = cos_sim(query_embedding, question_embeddings)
        top_idx = scores.argmax()
        return self.df.iloc[top_idx.item()]["Answer"]

# Flask app setup
app = Flask(__name__)

# Greetings list
greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "good night"]

def is_greeting(text):
    return any(word in text for word in greetings)

@app.route('/')
def home():
    return render_template('web1.html')

@app.route('/get', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()

    if is_greeting(user_input):
        return jsonify({"response": "Hello! How can I assist you with admissions today?"})

    try:
        query_embedding = chatbot.model.encode(user_input, convert_to_tensor=True)
        question_embeddings = chatbot.model.encode(chatbot.df["Preprocessed_Question"].tolist(), convert_to_tensor=True)
        scores = cos_sim(query_embedding, question_embeddings)
        max_score = torch.max(scores).item()

        if max_score < 0.65:
            reply = "I only answer questions related to admissions!!! Ask something related 😊"
        else:
            top_idx = scores.argmax()
            reply = chatbot.df.iloc[top_idx.item()]["Answer"]

    except Exception:
        reply = "Something went wrong while processing your request. Please try again."

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

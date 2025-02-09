# AI-Powered Health Assistant

## 📌 Project Overview
This project focuses on developing an **AI-powered health assistant** that provides **personalized health advice** using **natural language processing (NLP)**. The system aims to bridge the gap between **AI advancements** and **accessible healthcare guidance**, offering real-time, bilingual support for users.

## 🚀 Features
- 🏥 **Instant Health Advice** based on user queries
- 🗣 **Speech-to-Text & Text-to-Speech** for better accessibility
- 🌎 **Bilingual Support** (English & Hindi)
- ⚠ **Emergency Keyword Detection** for critical situations
- 📂 **Session-Based History Tracking** for personalized recommendations
- 📊 **Dynamic Diet and Wellness Planning** based on user preferences
- 🤖 **AI-Powered Chatbot** for interactive and engaging health discussions
- 📡 **Cloud-Based Deployment** for scalability and accessibility
- 🏋 **Integration with Fitness APIs** for real-time health tracking
- 🔍 **Symptom Checker** using AI-driven analytics

## 🔧 Technologies Used
- **Programming Language:** Python 3.11+
- **Frontend:** Streamlit (for UI)
- **AI Models:** Ollama LLM, Hugging Face embeddings
- **Retrieval Mechanism:** ChromaDB
- **Speech Processing:** Google Speech Recognition, gTTS
- **Libraries:** LangChain, NumPy, Pandas, FPDF
- **Deployment:** AWS/GCP, Docker (Optional)

## 📜 Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Future Enhancements](#future-enhancements)
5. [Contributors](#contributors)
6. [License](#license)

## 📖 Introduction
The AI-powered health assistant is designed to **help individuals make informed health decisions** by providing **accurate, region-specific, and personalized health recommendations**. It leverages **machine learning models, speech recognition, and retrieval-augmented generation (RAG)** to enhance user experience.

## 🛠 Installation
1. Ensure you have the necessary directories:
   ```bash
   mkdir -p data vectorstore
   ```
2. Clone the repository:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-health-assistant.git
   cd ai-health-assistant
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ingest data for Ollama 3.2:
   ```bash
   python ingest.py
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 💡 Usage
- **Enter a health-related query** (e.g., "What foods help with high blood pressure?")
- **Use voice input** for a hands-free experience
- **Receive AI-generated health advice** in English or Hindi
- **Get emergency alerts** when critical keywords (e.g., "heart attack symptoms") are detected
- **Check symptoms** and receive preliminary AI-based analysis
- **Generate a personalized wellness plan** based on dietary preferences and fitness levels

## 🔮 Future Enhancements
- 🔄 **Integration with Wearable Devices** (Smartwatches, fitness trackers)
- 🧠 **Advanced AI Models** for more precise recommendations
- 📌 **User Feedback Mechanism** to improve accuracy
- 🌏 **Multilingual Expansion** beyond English & Hindi
- 📈 **Real-Time Health Insights Dashboard**

## 🤝 Contributors
- **Rudra Kadel** (Project Lead)
- **Adharsh P.** (Mentor)

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---
✅ *Feel free to contribute by submitting issues or pull requests!*

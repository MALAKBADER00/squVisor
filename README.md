# SquVisor 🧠💬

SquVisor is a RAG-based chatbot designed to answer questions related to SQU (Sultan Qaboos University) regulations and policies. Built with Python and Django, it uses Retrieval-Augmented Generation (RAG) techniques to provide accurate and context-aware responses.

---

## 🚀 Features

- Intelligent chatbot for SQU-related queries  
- Uses OpenAI's GPT model via API  
- Retrieval-Augmented Generation (RAG) powered  
- Django web framework  

---

## ⚙️ Setup & Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/MALAKBADER00/squVisor.git
```
### 2. Create and activate Virtual Environment 
```
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies 
```
pip install -r requirements.txt
```
### 4. Create a `.env` File
Create a file named .env in the root directory of your project and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```
### 5. Start the Django Development Server
```bash
cd squ_visor
python manage.py runserver
```


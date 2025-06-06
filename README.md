# 🧠 Retrieval-Augmented Generation (RAG) PDF Demo

This is the companion code for my YouTube video where we walk through a basic RAG pipeline using Python and Flask.

## 🚀 Demo
Try it live: https://aishepherd.dev/portfolio/RAG/results

## 📹 Video
Watch the full video here: https://www.youtube.com/watch?v=BREQbLiRZFI

## 💡 Features
- Upload a PDF
- Chunk the document
- Embed with OpenAI
- Query using RAG-style prompting
- Return answers grounded in your content with sources

## 🛠️ Setup

```bash
git clone https://github.com/The-AI-Shepherd/RAG-demo.git
cd rag-demo
pip install -r requirements.txt
cp .env  # Add your OpenAI API key
python app.py


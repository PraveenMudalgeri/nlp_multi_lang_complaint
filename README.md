# Multilingual NLP-Based Complaint Processing and Local Language Generation System

## 📌 Overview

This project implements a multilingual NLP pipeline that accepts user complaints in any language and converts them into a structured, formal complaint in a selected regional language (Kannada or Hindi).

The system uses translation and basic NLP techniques to extract key information and generate a well-formatted complaint suitable for official use.

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| Frontend | React (Vite + TypeScript) |
| Styling | Tailwind CSS |
| External API | Sarvam AI (Translation & Language Processing) |

---

## 🧠 Features

- Multilingual input support (English, Hindi, Kannada, mixed)
- Automatic language detection
- Translation using Sarvam AI API
- Complaint type & location extraction
- Structured complaint generation
- Output in regional language (Kannada / Hindi)
- PDF download of formatted complaint
- Clean and minimal UI

---

## 🏗️ Project Structure

```
project/
├── app/
│   ├── main.py
│   ├── routes/
│   │   └── complaint.py
│   ├── schemas/
│   │   └── complaint.py
│   └── services/
│       ├── extractor.py
│       ├── formatter.py
│       └── sarvam.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ComplaintForm.tsx
│   │   │   └── OutputBox.tsx
│   │   ├── api.ts
│   │   ├── App.tsx
│   │   ├── index.css
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── .env.example
└── requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_api_key_here
SARVAM_BASE_URL=https://api.sarvam.ai
```

---

## 🚀 Backend Setup

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend server:

```bash
uvicorn app.main:app --reload
```

Backend runs at: `http://localhost:8000`

---

## 💻 Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## 🔌 API Reference

### `POST /process-complaint`

**Request Body**

```json
{
  "complaint_text": "The road near MG Road Bengaluru is full of potholes and causing issues",
  "target_language": "kn"
}
```

**Response**

```json
{
  "detected_language": "en",
  "complaint_type": "road",
  "location": "MG Road Bengaluru",
  "final_output": "..."
}
```

---

## 🔄 System Workflow

```
User Input (any language)
        ↓
Language Detection
        ↓
Translation to English (pivot language)
        ↓
Complaint Detail Extraction (type + location)
        ↓
Structured Complaint Generation
        ↓
Translation to Target Regional Language
        ↓
PDF Download
```

---

## 📄 Output Format

The generated complaint follows a formal structure:

- **Subject** — Brief description of the issue
- **Greeting** — Formal salutation
- **Issue Description** — Detailed explanation of the complaint
- **Request for Action** — What the complainant is seeking
- **Closing** — Formal sign-off

---

## ⚠️ Notes

- A valid Sarvam AI API key is required
- An active internet connection is needed for API calls
- PDF generation requires proper Unicode font support for regional scripts

---

## 🎯 Future Improvements

- Advanced NER for more accurate location extraction
- Support for additional regional languages
- Voice input integration
- Real-time translation feedback
- User authentication and complaint tracking

---

## 👨‍💻 Author

Developed as part of an NLP microproject focusing on multilingual processing and practical application of NLP pipelines.

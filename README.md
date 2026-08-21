# Enterprise RAG Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** system built in Python for answering questions from enterprise documentation.

The system focuses on **reliable, grounded responses** by combining semantic retrieval, adaptive query transformation, answerability detection, and local LLM inference.

> **Core principle:** Only answer when the retrieved documentation contains sufficient evidence. Otherwise, abstain.

---

## 🚀 Key Features

* Semantic vector retrieval
* Confidence-based adaptive retrieval
* Query transformation for ambiguous queries
* Weighted Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Answerability detection
* Grounded answer generation
* Explicit abstention for unsupported questions
* Local LLM inference using Ollama
* Retrieval and end-to-end evaluation

---

## 🏗️ Architecture

```text
User Query
    ↓
Initial Retrieval
    ↓
Confidence Check
    ↓
Low Confidence?
   ↙         ↘
 Yes          No
  ↓            ↓
Query        Continue
Rewrite
  ↓
Second Retrieval
  ↓
Weighted RRF
    ↓
Top-K Documents
    ↓
Answerability Detection
   ↙              ↘
Answerable       Not Answerable
    ↓                  ↓
Grounded Answer      Abstain
```

---

## 🔄 How It Works

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Qdrant Vector Store
    ↓
Semantic Retrieval
    ↓
Adaptive Query Transformation (if needed)
    ↓
Weighted RRF + Reranking
    ↓
Answerability Check
    ↓
Grounded Answer / Abstain
```

### 1. Document Processing

Enterprise documents are split into semantic chunks and converted into vector embeddings using **Sentence Transformers**.

These embeddings are stored locally in **Qdrant**.

### 2. Adaptive Retrieval

The system calculates a confidence margin between the top retrieval results:

```text
margin = top_score - second_score
```

If the retrieval confidence is low, the query is rewritten using a local LLM and retrieved again.

### 3. Weighted RRF

The original and transformed query results are combined using **Weighted Reciprocal Rank Fusion**.

```text
Original Query Weight    = 1.0
Transformed Query Weight = 1.5
RRF k                    = 60
```

### 4. Answerability Detection

Before generating an answer, the system checks whether the retrieved context actually contains enough information.

This prevents:

* Unsupported assumptions
* Missing-information hallucinations
* Incorrect policy inference

If the documentation does not support an answer:

```text
I don't have enough information in the provided handbook to answer that.
```

### 5. Grounded Generation

The LLM is instructed to:

* Use only retrieved context
* Avoid outside knowledge
* Never invent policies or facts
* Preserve important numbers and conditions
* Answer concisely

---

## 🛠️ Technology Stack

| Component            | Technology             |
| -------------------- | ---------------------- |
| Language             | Python                 |
| Embeddings           | Sentence Transformers  |
| Vector Database      | Qdrant                 |
| Retrieval            | Semantic Vector Search |
| Query Transformation | Ollama / Llama 3.2 3B  |
| Reranking            | Cross-Encoder          |
| Answerability        | Local LLM Judge        |
| Generation           | Ollama / Llama 3.2 3B  |
| Evaluation           | Custom Python Scripts  |

---

## 📁 Project Structure

```text
ent-rag-ast/
│
├── app/
│   ├── adaptive_retriever.py
│   ├── answerability.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── ingest.py
│   ├── query_transformer.py
│   ├── rag.py
│   ├── reranker.py
│   ├── retriever.py
│   └── vector_store.py
│
├── data/
│   ├── company_handbook.md
│   └── evaluation_questions.json
│
├── qdrant_data/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ent-rag-ast
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

Install Ollama, then pull the required model:

```bash
ollama pull llama3.2:3b
```

Test the model:

```bash
ollama run llama3.2:3b
```

---

## 📥 Build the Vector Store

Run the ingestion pipeline:

```bash
python -m app.ingest
```

This process:

```text
company_handbook.md
        ↓
     Chunking
        ↓
    Embeddings
        ↓
      Qdrant
```

The local vector database is stored in:

```text
qdrant_data/
```

---

## ▶️ Run the RAG Assistant

```bash
python -m app.rag
```

Example:

```text
Question:
How many vacation days do I get?

Answer:
You are entitled to 24 paid vacation days per calendar year.
```

---

## 📊 Evaluation

The project includes evaluation scripts for:

```bash
python -m app.evaluate_retrieval
python -m app.evaluate_query_transformation
python -m app.evaluate_adaptive_retrieval
python -m app.evaluate_answerability
python -m app.evaluate_full_pipeline
```

Key evaluation metrics include:

* Recall@1
* Recall@3
* MRR
* Answerable Detection Rate
* Negative Rejection Rate
* False Positive / Negative Rate

### Best Observed Adaptive Retrieval Result

```text
Recall@1: 100.00%
Recall@3: 100.00%
MRR:      100.00%
```

---

## 🧪 Example Behavior

| Query Type              | Example                                    | Expected Behavior        |
| ----------------------- | ------------------------------------------ | ------------------------ |
| Supported               | How many vacation days do I get?           | Grounded answer          |
| Semantic Paraphrase     | How often can employees work remotely?     | Correct policy retrieval |
| Unsupported             | How many sick leaves can employees take?   | Abstain                  |
| Missing Specific Detail | Does the company provide dental insurance? | Abstain                  |

---

## 🎯 Why This Is More Than Basic RAG

A basic RAG pipeline is:

```text
Query → Embedding → Vector Search → LLM
```

This project extends it with:

```text
Query
  ↓
Initial Retrieval
  ↓
Confidence Check
  ↓
Query Transformation
  ↓
Weighted RRF
  ↓
Reranking
  ↓
Answerability Detection
  ↓
Grounded Generation
  ↓
Answer / Abstain
```

This makes the system more robust against **ambiguous queries, retrieval errors, and hallucinated answers**.

---

## 🔮 Future Improvements

* Hybrid BM25 + vector retrieval
* Larger evaluation datasets
* Citation-aware answers
* Metadata filtering
* Retrieval latency monitoring
* Streaming responses
* FastAPI deployment
* Authentication and authorization
* Conversation memory
* CI/CD and automated regression testing
* Production observability

---

## 📌 What This Project Demonstrates

This project demonstrates practical AI engineering concepts including:

* RAG pipeline design
* Document chunking and embeddings
* Vector search with Qdrant
* Adaptive retrieval
* Query transformation
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Answerability detection
* Hallucination prevention
* Grounded LLM generation
* Local LLM inference
* Retrieval and end-to-end evaluation

> The goal is not simply to generate plausible answers, but to build a **reliable enterprise RAG system that retrieves the right evidence, verifies whether the evidence is sufficient, and abstains when the documentation does not support an answer.**

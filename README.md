<p align="center">
  <img src="media/demo.gif" alt="Demo" width="600"/>
</p>

# 🧠 Beauty & Electronics Chat Assistant (RAG-based LLM)

A domain-specific **AI shopping assistant** for Beauty and Electronics products, built using a full **Retrieval-Augmented Generation (RAG)** pipeline.

The system understands user shopping queries like:

> *“Recommend a facial cleansing tool under 20€”*
> *“Best earphones under 20€”*
> *“Find a good camera for wedding photography”*

It then retrieves real product information from the dataset and generates natural, grounded answers using a local LLM.

---

# 🚀 Features

### **🔍 Retrieval-Augmented Generation (RAG)**

* Vector search over product corpus (FAISS)
* Accurate chunk-based retrieval using **token-based splitting**
* Domain-aware (Beauty/Electronics) product matching
* Supports mixed-domain queries

---

### **🧠 Smart Query Parsing**

* Extracts:

  * **Price constraints:** “under 30”, “between 50–100”, “above 20”
  * **Rating constraints:** “above 4.3 stars”, “4+ rated”
  * **Desired number of products:** “recommend 3 serums”
* Domain detection using keyword lists
* Domain routing to Beauty/Electronics FAISS indices

---

### **⚡ Fast & Accurate Embeddings**

* Model: **BAAI/bge-small-en-v1.5** (preferred)
* Embeddings normalized for cosine similarity
* Fast GPU encoding via SentenceTransformers

---

### **🤖 Local LLM Generation**

* Default: **Mistral-7B-Instruct-v0.2** (quantized)
* Optional: FLAN-T5-Large baseline
* Structured shopping-assistant prompt
* Generates concise, grounded recommendations
* Each result includes:

  * Product name
  * Price
  * Rating
  * Reason for recommendation

---

### **🌐 Web App**

* Built with Flask
* Includes:

  * Input box for user query
  * Model-generated recommendations
  * `/api/search` JSON endpoint
  * Caching layer for repeated queries

---

# 🧩 Architecture

```
User Query
   │
   ▼
[ Query Parser ]
   │— Extracts domain, price filters, rating filters, requested count
   ▼
[ Embed Query with BGE-small ]
   ▼
[ FAISS Retrieval ]
   │— Beauty index
   │— Electronics index
   │— Global index
   ▼
Top-k Product Chunks
   ▼
[ LLM Generator: Mistral 7B ]
   │— Builds product-aware prompt
   ▼
Final Answer
```

---

# 🛠 Tech Stack

| Component    | Choice                               |
| ------------ | ------------------------------------ |
| Language     | Python                               |
| Embeddings   | BGE-small (`BAAI/bge-small-en-v1.5`) |
| Vector Store | FAISS                                |
| Chunking     | tiktoken (`cl100k_base`)             |
| LLM          | Mistral-7B-Instruct-v0.2 (quantized) |
| Parsing      | spaCy                                |
| Web          | Flask                                |
| Storage      | Parquet + joblib                     |

---

# 📁 Project Structure

```
project/
│
├── data/
│   ├── raw/
│   ├── worked/
│   │   ├── cleaned_full_corpus.parquet
│   │   ├── cleaned_corpus_rich.parquet
│   │   ├── chunks.parquet
│   └── tokenized/
│       ├── faiss_all.index
│       ├── faiss_beauty.index
│       ├── faiss_electronics.index
│       ├── chunks_embeddings.joblib
│       └── meta.joblib
│
├── src/
│   ├── tokenization_embeddings.py
│   ├── retrieve.py
│   ├── generate_mistral.py
│   ├── generate_flan.py
│   ├── model_manager.py
│   ├── paths.py
│   └── testing_querys.py
│
├── app.py
├── templates/
│   └── index.html
└── README.md
```

---

# 🔪 Data Processing

### **1. Cleaning**

Each record is normalized, cleaned, and merged into a single text field:

```python
parts = [
    f"Title: {row['title']}",
    f"Features: {row['features']}",
    f"Description: {row['description']}"
]
combined_text = ". ".join(str(p) for p in parts if p)
```

---

### **2. Chunking**

Token-based chunking using tiktoken:

* `chunk_size=200`
* `chunk_overlap=30`

Each chunk contains:

* doc_id
* chunk_id
* domain
* price
* rating
* categories
* title
* text (chunk body)

---

### **3. Embeddings**

Using **BAAI/bge-small-en-v1.5**:

* Fast & accurate for semantic search
* Embeddings normalized
* batched GPU encoding
* saved as numpy array via joblib

---

### **4. FAISS Indices**

Three FAISS indices:

* **faiss_all.index**
* **faiss_beauty.index**
* **faiss_electronics.index**

Plus domain index maps stored in `meta.joblib`.

---

# 🔍 Retrieval Logic

### Steps:

1. Detect domain(s)
2. Parse price/rating filters
3. Encode user query
4. Search domain-specific FAISS index
5. Apply filters
6. Fallback if low recall
7. Return top N products with metadata

Supports:

* Mixed-domain queries
* Numeric filtering
* Domain balancing
* De-duplication

---

# 🤖 LLM Generation

Mistral-7B-Instruct-v0.2 (quantized):

* Loaded with `ModelManager`
* Device-mapped automatically
* Builds product-aware prompt:

```
You are a shopping assistant...
Below are the relevant products...
Recommend exactly N...
```

Generates coherent, grounded product recommendations.

---

# 🌐 Web App (Flask)

Routes:

| Route         | Description        |
| ------------- | ------------------ |
| `/`           | User interface     |
| `/api/search` | JSON API           |
| `/health`     | Model health check |

Includes:

* Query caching
* FAISS retriever instance
* Mistral generator instance

---

# 🧪 Example Query

```
recommend a facial cleansing tool under 20
```

LLM output example:

> **Top options under 20€:**
> • Based on your request for a facial cleansing tool under 20 euros, I would recommend Product 3: Facial Cleansing Pads, Silicone Face Scrubbers Soft and Gentle....

---

# 📈 Evaluation

### Qualitative

* Run `testing_querys.py`
* Inspect retrieval logs
* Inspect generated answers

### Optional Quantitative

* Manually annotate relevant products for 20–50 queries
* Compute **Recall@K** for retrieval

---

# 🛑 Limitations

* No review-based embeddings yet
* Domain detection is keyword-based
* Mistral 7B may hallucinate with poor prompts
* Chunk size affects recall—needs tuning

---

# ✔ Installation

```
pip install -r requirements.txt
```
#### Download Datasets
Beauty dataset here (data\raw_meta_All_Beauty)
* [Beauty dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw_meta_All_Beauty/full-00000-of-00001.parquet)

Electronics dataset here (data\raw_meta_Electronics)
* [Electronics dataset 1](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw_meta_Electronics/full-00000-of-00010.parquet)
* [Electronics dataset 2](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/raw_meta_Electronics/full-00001-of-00010.parquet)

---

# ✔ Build Indices

```
python -m src.tokenization_embeddings
```

---

# ✔ Run App

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```
---

# 🤝 Contributing

Contributions are welcome!
If you’d like to improve the retrieval pipeline, add new models, optimize the web UI, or fix bugs, feel free to submit a Pull Request or open an issue.

---

# 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

# 👨‍💻 Author

**Abdul Haseeb**

* GitHub: [@Haseeb2510](https://github.com/Haseeb2510)
* LinkedIn: [Abdul Haseeb](https://www.linkedin.com/in/abdul-haseeb-172542243)

---

---

# 🎉 Acknowledgments

This project was made possible thanks to the open-source community and the powerful tools that support modern NLP, vector search, and LLM development.

Special thanks to:

* **MCAuley Lab (UC San Diego)** — for maintaining the Amazon product review and metadata datasets, which enabled high-quality product information retrieval for this project. Their long-standing research contributions to recommendation systems and product modeling made this work possible.
* **BAAI Research** — for the *BGE embedding models* used in semantic retrieval.
* **Mistral AI** — for the *Mistral-7B-Instruct* model powering natural-language recommendations.
* **SentenceTransformers team** — for the embedding framework used to encode product chunks efficiently.
* **FAISS (Meta AI)** — for the high-performance vector indexing library.
* **spaCy** — for fast, reliable NLP parsing and token extraction.
* **tiktoken** — for efficient tokenization used in the chunking stage.
* **Flask** — for the lightweight and flexible web framework powering the UI and API.
* **Pandas, NumPy, and PyArrow** — for powering all data cleaning, storage, and transformations.
* **The open-source ecosystem** — for maintaining all the tools, libraries, and models that made this end-to-end RAG system accessible, reproducible, and high-quality.

Grateful to the entire community for enabling this project.


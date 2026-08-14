# Production-Quality Hybrid AI-Driven E-Commerce Recommendation System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/SentenceTransformers-MiniLM--L6--v2-green.svg)](https://www.sbert.net/)
[![ResNet50](https://img.shields.io/badge/Torchvision-ResNet50-red.svg)](https://pytorch.org/vision/stable/models.html)
[![Flask REST API](https://img.shields.io/badge/Flask-REST--API-black.svg)](https://flask.palletsprojects.com/)
[![Test Suite](https://img.shields.io/badge/Pytest-42%20Passed-brightgreen.svg)](https://docs.pytest.org/)

A production-grade, multi-modal **Hybrid E-Commerce Recommendation & Semantic Search System** built using Python, Machine Learning, NLP (TF-IDF & Sentence Transformers), Computer Vision (ResNet50 CNN Feature Extraction), Item-Item Collaborative Filtering, Configurable Score Fusion, Transparent Explainable AI (XAI), a Flask REST API, and an Interactive Glassmorphic Web Interface.

---

## Architecture Overview

```mermaid
flowchart TD
    User([User / Browser UI]) <-->|HTTP / REST JSON| API[Flask REST API Server]
    
    subgraph Service[Singleton ML Recommendation Service]
        API <--> ServiceMgr[RecommendationService Manager]
        
        subgraph Models[Multi-Modal ML Signal Engines]
            CB[Content-Based TF-IDF Recommender]
            CF[Item-Item Collaborative Filtering]
            NLP[SentenceTransformer 384D Semantic Search]
            CNN[ResNet50 CNN 2048D Visual Feature Extractor]
        end
        
        ServiceMgr -->|Candidate Union| CG[Candidate Generation]
        CG -->|Raw Scores| Norm[Min-Max Score Normalization]
        Norm -->|Normalized Scores| Fusion[Configurable Weighted Fusion]
        Fusion -->|Ranked Candidates| XAI[Transparent Explainable AI Engine]
    end
    
    XAI -->|Ranked Recs + Signal Breakdown| API
```

---

## Key Features

1. **TF-IDF Content-Based Filtering**: Unigram/Bigram term frequency vectorization over product title, category, and description attributes with Cosine Similarity matching.
2. **Item-Item Collaborative Filtering**: Predicts user ratings and preferences by building an item rating correlation matrix $R \in \mathbb{R}^{U \times I}$.
3. **SentenceTransformer NLP Semantic Search**: 384-dimensional dense sentence embeddings using `all-MiniLM-L6-v2` for natural-language search queries (e.g., *"comfortable wireless headphones for gaming"*).
4. **ResNet50 CNN Image Similarity**: Deep Convolutional Neural Network feature extraction truncated at `fc` layer, generating 2048-dimensional visual vectors for visual product search.
5. **Candidate Generation & Min-Max Normalization**: Two-stage retrieval filtering thousands of items down to top-$N$ candidates before scaling heterogeneous raw scores onto a standardized $[0.0, 1.0]$ interval.
6. **Configurable Weighted Fusion**: Dynamic score ensembling ($\sum w_j = 1.0$) with automatic weight re-normalization when user history or queries are absent.
7. **Transparent Explainable AI (XAI)**: Derives exact mathematical score contribution breakdown ($\text{Contribution}_j = w_j \cdot S_{\text{norm}, j}$) and generates human-readable reasoning.
8. **Flask REST API Backend**: Singleton model service architecture serving `/api/products`, `/api/search`, `/api/recommendations/hybrid`, `/api/recommendations/user/<id>`, and `/api/health`.
9. **Interactive Web Interface**: Single-Page Application (SPA) featuring dark mode, glassmorphism aesthetics, live search, product detail modals, and animated XAI progress bars.
10. **Automated Pytest Suite**: 42 unit and integration tests passing 100%.

---

## Tech Stack

* **Core Language**: Python 3.11
* **Machine Learning & Data Science**: Scikit-Learn, Pandas, NumPy, SciPy
* **Deep Learning & Computer Vision**: PyTorch, Torchvision (ResNet50)
* **Natural Language Processing**: Sentence-Transformers (`all-MiniLM-L6-v2`), Hugging Face Transformers
* **Backend REST API**: Flask, Flask-CORS, Gunicorn
* **Frontend Web UI**: Vanilla JavaScript (ES6+), HTML5, CSS3 (Glassmorphism design system)
* **Testing & Serialization**: Pytest, Joblib

---

## Project Structure

```
hybrid-ai-recommender/
├── data/
│   ├── raw/                      # Raw products catalog & user interaction logs
│   ├── processed/                # Preprocessed clean_products.csv dataset
│   └── images/                   # Product catalog images
├── models/                       # Joblib serialized model artifacts
│   ├── content_based.joblib
│   ├── collaborative.joblib
│   ├── nlp_search.joblib
│   ├── image_features.joblib
│   └── hybrid_recommender.joblib
├── notebooks/                    # Analysis scripts & visual figure generators
│   ├── 01_exploratory_data_analysis.py
│   ├── 02_recommendation_analysis.py
│   ├── 03_semantic_and_image_analysis.py
│   └── 04_hybrid_recommendation_and_xai.py
├── reports/
│   └── figures/                  # 14 diagnostic visual evaluation plots (.png)
├── src/
│   ├── api/                      # Flask REST API & Web UI
│   │   ├── app.py                # Flask application factory
│   │   ├── service.py            # Singleton ML Service Manager
│   │   ├── routes/               # Modular route blueprints (health, products, search, recs)
│   │   └── static/               # Production Web UI (index.html, css/style.css, js/app.js)
│   ├── data/                     # Loader, schema mapper & image utilities
│   ├── preprocessing/            # Imputation, normalization & deduplication pipeline
│   ├── models/                   # ML Model implementations (Content, Collab, NLP, CNN, Hybrid, XAI)
│   ├── evaluation/               # Precision@K, Recall@K, NDCG@K ranking metrics
│   └── utils/                    # Centralized logging module
├── tests/                        # 42 Automated unit and integration tests
│   ├── test_loader.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   ├── test_evaluation.py
│   ├── test_nlp_search.py
│   ├── test_image_similarity.py
│   ├── test_hybrid_recommender.py
│   ├── test_explainability.py
│   └── test_api.py               # Flask REST API TestClient tests
├── Dockerfile                    # Production containerization setup
├── render.yaml                   # Render Cloud deployment specification
├── requirements.txt              # Project dependencies
└── README.md                     # Production documentation & educational guide
```

---

## Installation & Setup Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/user/hybrid-ai-recommender.git
cd hybrid-ai-recommender

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Steps 1–4)
```bash
# Step 1: Preprocessing & Data Validation
python -m src.preprocessing.pipeline
python -m src.data.validation

# Step 2: Content & Collaborative Models
python notebooks/02_recommendation_analysis.py

# Step 3: NLP Semantic Embeddings & CNN Image Feature Extraction
python notebooks/03_semantic_and_image_analysis.py

# Step 4: Hybrid Recommender, XAI & Ablation Study
python notebooks/04_hybrid_recommendation_and_xai.py
```

### 3. Launch Backend REST API Server & Web UI
```bash
python -m src.api.app
```
Access the interactive web UI at: **`http://127.0.0.1:5000`**

### 4. Run Pytest Test Suite (All 42 Unit & Integration Tests)
```bash
python -m pytest tests/ -v
```

---

## Flask REST API Endpoint Documentation

| Method | Endpoint | Description | Query / Body Parameters | Sample Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | System health check & model status | None | `{"status": "ok", "models_loaded": true}` |
| `GET` | `/api/products` | Paginated product catalog | `category`, `search`, `page`, `limit` | `{"data": [...], "pagination": {...}}` |
| `GET` | `/api/products/<id>` | Single product metadata | `product_id` in path | `{"product": {"product_id": "PROD_0101", ...}}` |
| `POST` | `/api/search` | SentenceTransformer vector search | `{"query": "headphones", "top_k": 10}` | `{"results": [...], "total_results": 10}` |
| `GET` | `/api/recommendations/user/<id>` | Personalized user hybrid recs | `user_id` in path, `top_k` query param | `{"user_id": "USER_001", "recommendations": [...]}` |
| `GET` | `/api/recommendations/product/<id>` | Similar product hybrid recs | `product_id` in path, `top_k` query param | `{"product_id": "PROD_0101", "recommendations": [...]}` |
| `POST` | `/api/recommendations/hybrid` | Full multi-signal hybrid scoring | `{"user_id": "...", "product_id": "...", "query": "..."}` | `{"recommendations": [{"final_hybrid_score": 0.85, ...}]}` |
| `GET` | `/api/recommendations/<id>/explanation` | XAI score attribution breakdown | `reference_product_id`, `query`, `user_id` | `{"explanation": {"score_contributions": {...}}}` |

---

## Evaluation Metrics & Ablation Study Results

### 1. Model Performance Comparison ($K = 5$)

| Recommendation Engine | Precision@5 | Recall@5 | NDCG@5 |
| :--- | :---: | :---: | :---: |
| **Content-Based (TF-IDF)** | 0.0000 | 0.0000 | 0.0000 |
| **Collaborative Filtering** | 0.0700 | 0.1167 | 0.0931 |
| **NLP Semantic Search** | 0.0171 | 0.0571 | 0.0386 |
| **Hybrid (Balanced Default)** | 0.0700 | 0.1167 | 0.0931 |
| **Hybrid (Collaborative-Heavy)** | **0.0800** | **0.1333** | **0.1065** |

### 2. Ablation Study

| Ablation Stage | Models Included | P@5 | R@5 | NDCG@5 |
| :--- | :--- | :---: | :---: | :---: |
| **Stage 1** | Content Only | 0.0000 | 0.0000 | 0.0000 |
| **Stage 2** | Content + Collaborative | **0.0800** | **0.1333** | **0.1065** |
| **Stage 3** | Content + Collaborative + NLP | 0.0700 | 0.1167 | 0.0931 |
| **Stage 4** | Full Hybrid (Content + Collab + NLP + Image) | 0.0700 | 0.1167 | 0.0931 |

---

## Educational Learning Breakdown: From Data to Web UI

### 1. Dataset Generation & Ingestion
* **What it does**: Ingests catalog metadata and user interaction ratings. Generates a realistic 140-product benchmark dataset if raw files are absent.
* **Why we need it**: ML models require structured tabular input.
* **Input/Output**: Raw CSV $\rightarrow$ Pandas DataFrames (`clean_products.csv`, `user_interactions_raw.csv`).
* **Concept**: Data persistence & schema contract enforcement.

### 2. Preprocessing & Validation Pipeline
* **What it does**: Imputes missing values, cleans text (lowercasing, HTML stripping), deduplicates products, and validates numeric bounds ($0 \le \text{price}$, $0 \le \text{rating} \le 5$).
* **Why we need it**: Garbage in, garbage out. Uncleaned text or out-of-bound ratings break vector similarity math.
* **Concept**: Data quality assurance & defensive preprocessing.

### 3. Content-Based TF-IDF Filtering
* **What it does**: Vectorizes title, category, and description text using Unigram/Bigram TF-IDF, computing pairwise Cosine Similarity.
* **Why we need it**: Recommends items with similar textual properties to reference products without requiring user history.
* **Concept**: TF-IDF (Term Frequency-Inverse Document Frequency) & Cosine Similarity in vector space.

### 4. Item-Item Collaborative Filtering
* **What it does**: Pivots interaction logs into a User-Item rating matrix $R \in \mathbb{R}^{U \times I}$, calculates item-item rating correlations, and predicts scores for unrated items.
* **Why we need it**: Captures implicit user preference patterns ("Users who liked A also liked B").
* **Concept**: Collaborative Filtering & Memory-Based Rating Prediction:
  $$\hat{r}_{u, i} = \frac{\sum_{j} S_{i,j} \cdot r_{u,j}}{\sum_{j} |S_{i,j}|}$$

### 5. SentenceTransformer NLP Dense Semantic Search
* **What it does**: Encodes product text into 384-dimensional dense vectors using `all-MiniLM-L6-v2` transformer model.
* **Why we need it**: Enables natural-language query search beyond exact keyword matching (e.g. matching *"sneakers"* to *"running shoes"*).
* **Concept**: Transformer Attention Embeddings & Dense Vector Retrieval.

### 6. ResNet50 CNN Image Feature Extraction
* **What it does**: Passes product catalog images through a pretrained ResNet50 Convolutional Neural Network truncated at the final classification layer, extracting 2048-dimensional visual embeddings.
* **Why we need it**: Captures visual appearance (color, shape, pattern) for visual recommendation.
* **Concept**: Transfer Learning & Feature Vector Extractor Mode (`torch.no_grad()`).

### 7. Candidate Generation
* **What it does**: Union top-$N$ candidate items from active models, removes duplicates, and filters out already-purchased products.
* **Why we need it**: Avoids performing expensive multi-model scoring across millions of catalog items.
* **Concept**: Two-Stage Retrieval Architecture (Candidate Generation $\rightarrow$ Heavy Ranking).

### 8. Min-Max Score Normalization
* **What it does**: Rescales raw scores from heterogeneous distributions (ratings 1–5 vs similarity 0–1) onto $[0.0, 1.0]$.
* **Why we need it**: Prevents models with larger raw numerical scales from dominating weighted fusion.
* **Concept**: Feature Scaling & Score Standardization:
  $$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$

### 9. Configurable Weighted Fusion
* **What it does**: Computes final hybrid score: $\text{Hybrid Score} = \sum w_j \cdot S_{\text{norm}, j}$ ($\sum w_j = 1.0$).
* **Why we need it**: Ensembles all model strengths into a single unified recommendation score.
* **Concept**: Linear Ensemble Scoring & Dynamic Re-weighting.

### 10. Transparent Explainable AI (XAI)
* **What it does**: Calculates exact signal score contribution ($\text{Contribution}_j = w_j \cdot S_{\text{norm}, j}$) and outputs natural-language bullet points.
* **Why we need it**: Builds user trust by explaining *why* an item was recommended.
* **Concept**: Transparent Score Attribution & Explainable Recommendation.

### 11. Singleton ML Service Manager
* **What it does**: Loads all fitted `.joblib` model artifacts once into RAM at server startup.
* **Why we need it**: Prevents reloading heavy models on every HTTP request.
* **Concept**: Singleton Pattern & Memory Caching.

### 12. Flask REST API Backend
* **What it does**: Exposes structured JSON REST endpoints with CORS, input validation, and HTTP status codes.
* **Why we need it**: Decouples recommendation logic from frontend UI presentation.
* **Concept**: RESTful Web Services & Separation of Concerns.

### 13. Interactive Web Frontend UI
* **What it does**: Provides a modern, responsive Single-Page Application (SPA) displaying products, live NLP search, user profile selection, and visual XAI signal progress bars.
* **Why we need it**: Delivers an intuitive end-user shopping experience.
* **Concept**: Single-Page Application & Responsive UX.

---

## 20 Technical Interview Preparation Q&As

### Category A: Project Architecture & Strategy (Questions 1–5)
1. **Q: Why did you build a hybrid recommendation system instead of using a single model?**
   * *A*: Single models have fundamental failure modes: Collaborative Filtering suffers from user cold-start; Content-Based lacks serendipity; NLP Search requires search text; CNN Vision requires image data. A hybrid ensemble combines all signals to deliver reliable recommendations across all user scenarios.
2. **Q: How did you structure the candidate generation pipeline?**
   * *A*: We implemented a two-stage retrieval pattern. The Candidate Generation stage retrieves the top-$N$ items from each active model, unions candidate IDs, removes duplicates, and excludes items the user has already rated. The Ranking stage normalizes and scores only this candidate pool.
3. **Q: How does your system handle Cold-Start users?**
   * *A*: When an unknown user ID is detected, the Collaborative Filtering signal is deactivated. Active weights are dynamically re-normalized across Content, NLP, and CNN signals, and popular/top-rated fallback products are returned.
4. **Q: How did you ensure high performance during API inference?**
   * *A*: We built a Singleton `RecommendationService` that pre-loads all serialized `.joblib` model artifacts into RAM at server startup. No models or feature matrices are reloaded during HTTP requests.
5. **Q: How is the backend decoupled from the frontend?**
   * *A*: The Flask REST API communicates strictly via standard JSON endpoints (`/api/products`, `/api/search`, `/api/recommendations/hybrid`). The frontend is a client SPA consuming these REST contracts.

### Category B: Machine Learning & Normalization (Questions 6–10)
6. **Q: Why can't raw model scores simply be added together?**
   * *A*: Raw scores operate on incommensurable distributions—Collaborative Filtering produces ratings from 1.0 to 5.0, while Content and NLP models produce cosine similarities from 0.0 to 1.0. Adding raw scores causes ratings to dwarf cosine similarities.
7. **Q: What score normalization technique did you use and why?**
   * *A*: We used Min-Max Normalization across the candidate pool: $S_{\text{norm}} = (S - S_{\min}) / (S_{\max} - S_{\min})$. It preserves relative scoring distance while bounding all model outputs strictly to $[0.0, 1.0]$.
8. **Q: How do you enforce valid hybrid weights?**
   * *A*: `validate_weights()` asserts that all weight values are non-negative and sum to $1.0 \pm 1e-4$, throwing an explicit `ValueError` if invalid.
9. **Q: What happens when an active model signal is missing (e.g. no search query provided)?**
   * *A*: The system identifies active signals, sets inactive scores to 0, and dynamically re-normalizes active signal weights so they sum to 1.0.
10. **Q: How did you evaluate the hybrid recommender system?**
    * *A*: We split user interaction logs into train/test sets and computed Precision@K, Recall@K, and NDCG@K ($K=5, 10$) across individual models, weight experiments, and ablation stages.

### Category C: NLP, Deep Learning & Vision (Questions 11–15)
11. **Q: Why did you choose SentenceTransformers over standard TF-IDF for semantic search?**
    * *A*: TF-IDF relies on exact term overlap (lexical matching). SentenceTransformers (`all-MiniLM-L6-v2`) encode text into 384-dimensional dense semantic vectors, capturing conceptual intent so queries like *"sneakers"* match *"running shoes"*.
12. **Q: How is the ResNet50 CNN model used for image similarity?**
    * *A*: We load pretrained ResNet50 weights, replace the final classification layer (`fc`) with `nn.Identity()`, and run forward inference in `torch.no_grad()` mode to extract 2048-dimensional feature vectors.
13. **Q: What limitation did you discover during image similarity investigation?**
    * *A*: Synthetic catalog images generated for offline testing shared category background colors. ResNet50 extracted background color features, creating high intra-category visual similarity ($\approx 0.71$). We set `image_weight = 0.15` to balance visual influence.
14. **Q: What is cosine similarity and how is it calculated?**
    * *A*: Cosine similarity measures the angle between two multi-dimensional vectors: $\text{CosineSim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$.
15. **Q: What is an Ablation Study and what did yours reveal?**
    * *A*: An ablation study evaluates performance as components are incrementally added. Ours proved that adding Collaborative Filtering to Content-Based matching yields the largest single accuracy gain ($P@5$ increased from 0.00 to 0.08).

### Category D: Explainable AI & System Integration (Questions 16–20)
16. **Q: How does your Explainable AI (XAI) engine work?**
    * *A*: We implement transparent score attribution where each signal's mathematical contribution ($\text{Contribution}_j = w_j \cdot S_{\text{norm}, j}$) is computed. Dominant contribution values trigger human-readable explanation bullet points.
17. **Q: Is your XAI system using SHAP or LIME?**
    * *A*: No. SHAP and LIME approximate non-linear black-box gradients. Our system uses exact, transparent linear score attribution derived directly from weighted fusion math.
18. **Q: What limitation exists in your XAI implementation?**
    * *A*: It provides transparent model attribution (explaining *how* the model calculated the score), but does not prove causal human intent or guarantee objective product perfection.
19. **Q: How do you handle HTTP errors in the Flask API?**
    * *A*: Global error handlers intercept 400, 404, and 500 exceptions, returning standardized JSON error objects without exposing raw Python stack traces.
20. **Q: How is the application prepared for cloud deployment?**
    * *A*: Containerized via `Dockerfile` using Python 3.11-slim and Gunicorn, accompanied by `render.yaml` for automated web service deployment.

---

## Deployment & Containerization Guide

### Run with Docker

```bash
# Build Docker image
docker build -t hybrid-ai-recommender .

# Run Docker container
docker run -d -p 5000:5000 --name recommender-app hybrid-ai-recommender
```
Access the application at `http://localhost:5000`.

---

## Final Project Status Report

* **Final Architecture**: Two-Stage Multi-Modal Hybrid Engine + Flask REST API + Single-Page Web UI.
* **Complete Project Structure**: 25 Python modules across `src/`, `notebooks/`, and `tests/`.
* **Backend Endpoints**: 8 REST endpoints fully operational under `/api/*`.
* **Frontend Features**: Dark mode glassmorphism UI, NLP search bar, user profile switcher, product modals, visual XAI progress bars.
* **ML Models**: Content-Based TF-IDF, Item-Item Collaborative Filtering, SentenceTransformer 384D, ResNet50 CNN 2048D.
* **Hybrid Formula**: Normalized Weighted Fusion ($\sum w_j \cdot S_{\text{norm}, j}$).
* **XAI Approach**: Transparent Score Attribution ($\text{Contribution}_j = w_j \cdot S_{\text{norm}, j}$).
* **Test Suite Status**: **42 / 42 Unit & Integration Tests Passed (100%)**.
* **Deployment Status**: Production-ready with `Dockerfile` and `render.yaml`.

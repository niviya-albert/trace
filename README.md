# 🛡️ TRACE — AI Reunification & Claim Verification Engine

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![HuggingFace Models](https://img.shields.io/badge/CLIP%20%2B%20MiniLM-100%25%20Offline-FFA116?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)

**"An AI reunification engine — built to find the match, and prove it belongs to the right person."**

**Team Grey Matter** • *MakersNeedMore (MnM) — Round 2*  
*Niviya Albert • Adithyan M J • Diya Paramanand*

[Overview](#-overview) • [Key Capabilities](#-key-capabilities--differentiators) • [Architecture](#-system-architecture) • [Scoring Engine](#-multi-modal-scoring-engine) • [Anti-Fraud System](#-anti-fraud-verification--handover) • [Evaluation Guide](#-evaluation--testing-guide) • [API Docs](#-api-endpoints-reference) • [Quick Start](#-quick-start)

---

</div>

## 📌 Overview

**TRACE** is an AI-powered operations and decision-support engine engineered specifically for high-throughput institutional Lost & Found counters — including **railway lost property offices, airport terminals, transit authorities, and university campus security**.

Traditional lost-and-found desks suffer from two major operational bottlenecks:
1. **Vocabulary & Perception Divergence**: Exact keyword search fails when passenger reports and desk staff use different wording (e.g., *"navy blue canvas backpack"* vs. *"black rucksack with dual straps"*).
2. **Claim Fraud & Erroneous Handoffs**: Desks lack an objective, privacy-preserving method to verify ownership without giving away distinguishing details to fraudulent claimants.

**TRACE solves both challenges through a closed-loop multi-modal fusion and anti-fraud verification pipeline.**

---

## ⚡ Key Capabilities & Differentiators

| Capability | Legacy Lost & Found Desks | TRACE AI Reunification Engine |
| :--- | :--- | :--- |
| **Search Mechanism** | Keyword/Exact Token Match | **4-Way Multi-Modal Fusion** (CLIP + MiniLM + Zone + Time) |
| **Vocabulary Mismatches** | ❌ **Fails (0 matches returned)** | ⚡ **Surfaces true match (~81% confidence)** |
| **Missing Photo Handling** | Manual triage / search failure | 🔄 **Dynamic Weight Re-Normalization** (50% Text, 25% Loc, 25% Time) |
| **Explainability (XAI)** | Black box or none | 💡 **Plain-Language AI Decision Driver** for operator trust |
| **Ownership Verification** | Subjective staff guesswork | 🔒 **Zero-Leakage Challenge Question & Deterministic Check** |
| **Handover Accountability** | Paper slips / unverified logs | 🧾 **Cryptographic Handover Audit Receipts** |
| **Network Resilience** | Cloud API dependencies | 🌐 **100% Offline Resilience** (Pre-warmed local embedding cache) |

---

## 🏗️ System Architecture

```
                                  +-----------------------+
                                  |   Staff Desk (Web)    |
                                  |   Streamlit Frontend  |
                                  +-----------+-----------+
                                              |
                                     (REST / JSON API)
                                              v
+-----------------------------------------------------------------------------------------+
|                                  FastAPI REST Backend                                   |
|                                                                                         |
|   • POST /found-items                • GET /found-items/{id}/matches   • POST /lost-reports |
|   • POST /found-items/{id}/challenge • POST /challenges/{id}/answer    • GET /dashboard/queue|
+---------------------------------------------+-------------------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
+---------------------------------------------+   +---------------------------------------+
|          Multi-Modal Scoring Engine         |   |        Anti-Fraud Verification        |
|                                             |   |                                       |
|  • Visual: CLIP ViT-B/32 (40%)              |   |  • Vaulted Hidden Attribute           |
|  • Text: MiniLM-L6-v2 Semantic (30%)        |   |  • Zero-Leakage Challenge Generator   |
|  • Location: Station Zone Matrix (15%)      |   |  • Deterministic Key-Term Matcher     |
|  • Time: 14-Day Linear Decay (15%)          |   |  • Automated Green / Red Decision     |
|  • Dynamic No-Photo Re-normalization        |   |  • Verifiable Handover Audit Receipt  |
|  • Side-by-Side Legacy Keyword Comparison   |   +---------------------------------------+
|  • Plain-Language AI Decision Driver        |
+----------------------+----------------------+
                       |
                       v
+---------------------------------------------+
|                SQLite Engine                |
|  • Found Items Table   • Lost Reports Table |
|  • Claim Challenges    • Seed Demonstrations|
+---------------------------------------------+
```

---

## 🔬 Multi-Modal Scoring Engine

TRACE fuses four distinct operational signals into a calibrated confidence score between **0.0 and 1.0 (0% - 100%)**:

### 1. Standard Weight Configuration (With Photo)
$$\text{Score}_{\text{fused}} = 0.40 \cdot S_{\text{visual}} + 0.30 \cdot S_{\text{text}} + 0.15 \cdot S_{\text{location}} + 0.15 \cdot S_{\text{time}}$$

- **Visual Similarity ($S_{\text{visual}}$ - 40%)**: `openai/clip-vit-base-patch32` image embeddings normalized and scored via Cosine Similarity.
- **Semantic Text ($S_{\text{text}}$ - 30%)**: `sentence-transformers/all-MiniLM-L6-v2` dense text embeddings capturing semantic intent regardless of exact vocabulary.
- **Location Zone Adjacency ($S_{\text{location}}$ - 15%)**: Station topology matrix evaluating proximity across platforms, concourses, commercial areas, train compartments, and parking zones.
- **Time Proximity ($S_{\text{time}}$ - 15%)**: 14-day linear decay window:
  $$S_{\text{time}} = \max\left(0, 1 - \frac{|\Delta t_{\text{days}}|}{14}\right)$$

### 2. Dynamic Weight Re-Normalization (No Photo on Lost Report)
When a passenger files a report without an image, TRACE smoothly re-normalizes weights without score degradation:
$$\text{Score}_{\text{fused}} = 0.50 \cdot S_{\text{text}} + 0.25 \cdot S_{\text{location}} + 0.25 \cdot S_{\text{time}}$$

### 3. Explainable AI (XAI) Decision Driver
Every match surfaces a plain-language summary explaining the primary driver of the match decision:
- *"Matched primarily on photo similarity, despite different wording."*
- *"Matched primarily on strong visual and semantic similarity."*
- *"Matched primarily on where and when it was found."*

---

## 🔒 Anti-Fraud Verification & Handover

To prevent fraudulent claims and mistaken handovers, TRACE implements a 3-step verification protocol:

```
[Staff Intake] ────> Vaults Secret Detail (e.g., "airline tag initials R.S.")
                           │
                           ▼
[Claimant Check] ──> Auto-Generates Challenge Question (Zero leakage)
                     "Does the item have any tags, stickers, or identifying markings?"
                           │
                           ▼
[Evaluation] ──────> Claimant: "It has my airline baggage tag with initials R.S."
                     Result: ✅ MATCH CONFIRMED (Matched: 'r.s.', 'baggage tag')
                           │
                           ▼
[Handover] ────────> Official Audit Receipt Generated with Cryptographic Signature
```

1. **Feature Vaulting**: Desk staff inputs physical characteristics plus one **non-public distinguishing feature** (e.g., internal tag, specific scratch, custom sticker, engraved date).
2. **Zero-Leakage Question Generation**: The engine formulates a generic challenge question that gives away no specifics to the claimant.
3. **Deterministic Keyword Verification**: The claimant's verbal or typed response is validated against key terms.
4. **Official Audit Receipt**: Successful verification unlocks the authorization button, generating a timestamped, signed handover record.

---

## 🎮 Evaluation & Testing Guide

TRACE includes pre-loaded interactive demo presets directly accessible from the Streamlit sidebar:

### Scenario 1: Mismatched Vocabulary Resolution
1. Open the **Staff Dashboard** at `http://localhost:8501`.
2. In the left sidebar, click **"🎒 Demo 1: Mismatched Wording"**.
3. Click **"🚀 Log Property & Run Multi-Modal Search"**.
4. In Screen 2 (**Multi-Modal Matches**):
   - Switch the toggle to **"🔍 Legacy Keyword Search"** → See **0 Matches** returned due to vocabulary mismatch (*"navy backpack"* vs *"black rucksack"*).
   - Switch to **"⚡ TRACE AI Multi-Modal Fusion"** → Lost Report #1 surfaces immediately with **~81% confidence**.
   - Note the **AI Decision Driver**: *"Matched primarily on photo similarity, despite different wording."*

### Scenario 2: Dynamic No-Photo Re-Normalization
1. In the sidebar, click **"⌚ Demo 2: Missing Photo Re-norm"**.
2. Submit the item and inspect the match breakdown in Screen 2.
3. Observe that visual weight is re-allocated to **50% Text, 25% Location, 25% Time** with clean badge indicators.

### Scenario 3: Anti-Fraud Verification (False vs. Genuine)
1. In the sidebar, click **"🧳 Demo 3: False Claim vs Genuine"**.
2. Submit the item and click **"🛡️ Proceed to Claim Verification"**.
3. Click **"❌ Load Fraudulent Answer"** (*"There is a standard blue ribbon tied around the handle"*) → Click **"🔍 Evaluate Claim Answer"** → Instant **Red Flag** (*Flagged for Staff Review*).
4. Click **"✅ Load Genuine Answer"** (*"It has an airline baggage tag with initials R.S."*) → Click **"🔍 Evaluate Claim Answer"** → Instant **Green Confirmation** (*Claim Verified*).
5. Click **"📦 Authorize & Confirm Physical Release"** to generate the **Official Handover Receipt**.

---

## 📁 Repository Structure

```
Trace/
├── backend/
│   ├── database.py       # SQLite connection and session management
│   ├── embeddings.py     # CLIP ViT-B/32 & MiniLM-L6-v2 local caching & inference
│   ├── main.py           # FastAPI application routes and dependency injection
│   ├── models.py         # SQLAlchemy ORM models (FoundItem, LostReport, ClaimChallenge)
│   ├── schemas.py        # Pydantic response and request validation schemas
│   ├── scoring.py        # 4-way multi-modal fusion, zone adjacency, time decay, & XAI driver
│   ├── seed.py           # Synthetic dataset and seed image generation (7 benchmark pairs)
│   └── verification.py   # Challenge generator & deterministic keyword matcher
├── frontend/
│   ├── app.py            # 3-step Streamlit operations desk UI & interactive comparison
│   └── styles.py         # Custom dark-mode design system with visual score bars & badges
├── data/
│   ├── seed/             # Seed JSONs and pre-generated image representations
│   └── uploads/          # Physical intake photo storage directory
├── check_system.py       # Automated 6-point live end-to-end integration health check
├── test_trace.py         # Unit & integration test suite (FastAPI TestClient & scoring checks)
├── run.py                # Single-command launcher for backend, frontend, and offline AI
├── requirements.txt      # Pinned dependency requirements
└── README.md             # Project documentation and evaluation guide
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/found-items` | Log found property with photo upload, location, time, and secret vaulted attribute |
| `GET` | `/found-items/{id}/matches` | Retrieve top-5 ranked candidate lost reports with detailed multi-modal score breakdowns |
| `POST` | `/lost-reports` | Register a new passenger lost report |
| `POST` | `/found-items/{id}/challenge` | Auto-generate a zero-leakage challenge question from vaulted attribute |
| `POST` | `/challenges/{id}/answer` | Submit claimant answer and receive deterministic green/red verification verdict |
| `GET` | `/dashboard/queue` | Retrieve active queue of open items with their highest-confidence match |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation UI |

---

## 🚀 Quick Start

### Prerequisites
- **Python**: Version `3.10` or higher
- **Hardware**: Standard CPU (PyTorch runs lightweight CPU inference locally in milliseconds)

### 1. Installation
Clone the repository and install required dependencies:
```bash
# Clone repository
git clone https://github.com/niviya-albert/TRACE.git
cd TRACE

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch TRACE (Single Command)
Run the automated bootstrapper that initializes the SQLite database, warms up offline AI embeddings, launches FastAPI on port `8000`, and starts Streamlit on port `8501`:
```bash
python run.py
```

- **Operations Dashboard**: [http://localhost:8501](http://localhost:8501)
- **Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Automated Testing

Execute the complete automated test suite covering embedding similarity, time decay calculations, weight re-normalization, challenge generation, and API endpoints:

```bash
python test_trace.py
```

Run live end-to-end operational health verification against running services:
```bash
python check_system.py
```

---

## 📄 License & Attribution

This project is developed by **Team Grey Matter** (*Niviya Albert, Adithyan M J, Diya Paramanand*) for **MakersNeedMore (MnM) — Round 2**. 

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <sub>Engineered for high-throughput, mission-critical lost property reunification operations.</sub>
</div>


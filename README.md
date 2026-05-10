# 👗 Dress Demand Prediction Engine
### Built for Maalde Fashion — AI/ML Hiring Assignment

---

**Full Name:** Devashish Rajeshrao Jore
**Mobile No:** +91-8320759063
**Email:** devashishjore0511@gmail.com

---

## What is this project?

When a fashion business designs a new dress, one of the hardest questions is:

> *"How many pieces should I manufacture?"*

Too few — you lose sales. Too many — you lose money on unsold stock.

This project solves exactly that. You upload a photo of a new dress design, and the system tells you approximately how many pieces it is likely to sell — based on patterns learned from 129 past designs and their real sales data.

---

## 🎬 Demo Video

[▶️ Click to Watch Demo](https://youtu.be/aPk7veoITWk)

## The Real Challenge — Connecting Images to Sales Data

Before any ML could happen, there was a fundamental data engineering problem to solve.

The images were named like this:
```
WhatsApp Image 2026-04-29 at 1.26.56 PM.jpeg
WhatsApp Image 2026-04-29 at 1.26.56 PM (1).jpeg
```

No product codes. No labels. Just WhatsApp export names.

The sales data had product codes like `500001`, `10029028` — but no connection to any image filename whatsoever.

After examining the images carefully, I found that every dress photo had the product code printed as a watermark at the bottom of the image itself:

```
KS_Kurta with Plazo & Dupatta-119 Fabric-Crepe Size-XL,2XL,M,L Dno-5_27_10022578
```

That `10022578` at the end is the product code.

So I built an **OCR-based automated pipeline** that:

1. Crops only the bottom 15% of each image — where the code always appears
2. Runs EasyOCR on that small strip — much faster than scanning the full image
3. Extracts any 6+ digit number from the detected text
4. Checks if that number exists in the sales CSV
5. If yes — match confirmed and saved. If no — flagged for manual review.

Out of 181 images, **119 matched automatically**. I manually verified and fixed 10 more. Final training dataset: **129 images with confirmed sales data.**

---

## Why OCR Didn't Match All 181 Images

This is worth being transparent about because it directly affected model training:

- **Watermark opacity** — some codes were semi-transparent and blended into the background
- **Decorative fonts** — stylized watermark text confused the OCR reader
- **Low contrast** — cream/off-white text on a light background is difficult for OCR to detect
- **Cropped images** — some photos were cropped and the bottom code strip was partially or fully missing
- **WhatsApp compression** — image compression blurred the fine print at the bottom

This left 52 images unmatched and excluded from training. With a larger confirmed dataset the model would generalize better. The root fix for this in a production system would be to store product codes in image filenames or metadata at the time of the photoshoot — eliminating the OCR dependency entirely.

---

## How the Prediction System Works

### Full Pipeline

```
181 Dress Images
      ↓
OCR Bottom Strip Extraction (EasyOCR + OpenCV crop)
      ↓
Product Code Matching with Sales CSV
      ↓
129 Confirmed Image-Sales Pairs
      ↓
MobileNetV2 Feature Extraction (1280 features per image)
      ↓
GradientBoosting Regressor Training
      ↓
Saved Model → demand_model.pkl
      ↓
Streamlit UI — Upload New Image → Predict Qty
```

### Why MobileNetV2 for Feature Extraction?

Training a CNN from scratch on 129 images would overfit immediately and produce unreliable predictions. MobileNetV2 is pre-trained on 1.2 million images and already understands colors, textures, embroidery patterns, fabric shine, and design complexity. I used it purely as a **feature extractor** — feed in a dress image, get back 1280 numbers describing its visual characteristics.

This is called **Transfer Learning** and is the standard industry approach for small image datasets.

| Option | Reason Not Selected |
|---|---|
| Train CNN from scratch | Needs 10,000+ images minimum |
| ResNet50 | Heavier, slower, overkill for this dataset size |
| EfficientNet | Good but slower to load and run |
| **MobileNetV2** | ✅ Lightweight, fast, accurate, ideal for small data |

### Why GradientBoosting Regressor?

After extracting 1280 features per image I trained and compared two models:

| Model | MAE (Mean Absolute Error) |
|---|---|
| RandomForest | 23.08 pieces |
| **GradientBoosting** | **13.83 pieces** ✅ |

GradientBoosting won clearly. It builds decision trees sequentially where each tree corrects the errors of the previous one — this works well when patterns are subtle and the dataset is small.

An MAE of 13.83 means predictions are off by roughly 14 pieces on average. For 129 training samples that is a reasonable and honest result.

---

## Must Answer Questions

### 1. How did you approach this problem?

The first step was understanding the data before writing any code. The images had no useful filenames and the sales CSV had no image references. The connection had to be found manually by inspecting the images — which revealed product codes embedded in the watermarks.

I built an OCR pipeline first to automate this mapping, then cleaned and aggregated the sales data (705 raw rows → 146 unique product codes), then merged both into a clean training dataset of 129 confirmed pairs.

Only after this data engineering phase did I move to modeling — using MobileNetV2 for feature extraction and GradientBoosting for regression. The full pipeline was built and trained on Google Colab using free GPU, and the final UI was built with Streamlit.

The honest ratio of time spent: roughly 70% data engineering, 30% modeling. That is realistic for production ML work.

### 2. How does your prediction system work?

A new dress image is uploaded through the Streamlit UI. MobileNetV2 extracts 1280 visual features from the image — capturing color, texture, embroidery density, fabric type, and overall design complexity. These features are passed into the trained GradientBoosting model which outputs a predicted sales quantity based on patterns learned from 129 historical image-sales pairs.

The UI displays the predicted quantity, a demand category (Low / Medium / High), and a comparison against average historical sales.

### 3. What patterns did you find in the data?

- **Code 500001 dominated with 1032 total qty across 57 orders** — the single bestselling design by a large margin. It is a heavily embroidered cream/off-white suit set priced around ₹1295.
- **Most designs had very low order counts** — 1 to 6 entries — suggesting the catalog is wide but volume per design is generally small.
- **Price range across all designs: ₹695 to ₹1650** — mid-range designs in the ₹1200–₹1400 band showed higher sales volumes.
- **Full sets with dupatta** appeared more frequently among higher-selling codes than standalone kurtas.
- **Sales data was concentrated on a single date (2026-04-22)** — suggesting this was a snapshot export rather than a full historical record.

### 4. Where can your system fail?

- **Small training set** — 129 samples is genuinely limited for visual ML. Predictions on design styles not well represented in training may be unreliable.
- **OCR matching gaps** — 52 images were excluded from training due to failed OCR matching. Those visual patterns are invisible to the model.
- **Price ignored** — the model only uses visual features. A beautiful dress priced too high will receive an overestimated prediction.
- **No seasonal awareness** — festive season demand vs off-season demand differs significantly. The current data has no time dimension.
- **Single date data** — most entries are from one date, so the model cannot learn trends over time.
- **New design styles** — if the business launches a completely new style that looks nothing like training data, predictions will be unreliable.

### 5. If you had more time, how would you improve this system?

- **More confirmed training data** — the single highest-impact improvement. 500+ matched image-sales pairs would significantly improve model performance.
- **Fix data collection at source** — embed product codes in image filenames at photoshoot time. This removes the OCR dependency completely and makes future data pipelines trivial.
- **Add price as a feature** — combining visual features with the product price point would make predictions much more realistic.
- **Manual design tags** — adding 5–6 tags per image (occasion type, fabric, color family, embroidery level) would give the model far more to learn from than raw pixels alone.
- **Time-based sales data** — a proper historical record across months and seasons would allow the model to learn seasonal demand patterns.
- **Prediction confidence intervals** — instead of "42 pieces" show "35–50 pieces (medium confidence)" to help the business make better manufacturing decisions.
- **Automatic retraining** — as new sales data comes in, the model should retrain on a schedule to stay current with shifting demand patterns.

---

## Project Structure

```
dress-demand-predictor/
│
├── data/
│   ├── sales_data.csv            # Raw sales data (705 rows)
│   ├── aggregated_sales.csv      # Cleaned sales (146 unique codes)
│   ├── ocr_mapping.csv           # Image to product code mapping
│   └── final_dataset.csv         # Final merged training dataset (129 rows)
│
├── models/
│   └── demand_model.pkl          # Trained GradientBoosting model
│
├── notebooks/
│   └── pipeline.ipynb            # Full Colab notebook (OCR + Training)
│
├── images/
│   └── sample_dress.jpg          # Sample image for testing the app
│
├── app.py                        # Streamlit prediction UI
├── clean_csv.py                  # Step 1 — cleans raw sales CSV
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

> **Note on full image dataset:** The complete 181 dress images are hosted on Google Drive provided by Maalde. Due to repository size constraints they are not included here.
> Drive link: https://drive.google.com/drive/folders/1PxLPhLtTpt1YR1hisvHeRZ1U-3sKV6EM

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/dress-demand-predictor.git
cd dress-demand-predictor
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## How to Use the App

1. Open the app in your browser
2. Click **"Upload Dress Photo"**
3. Select any dress image (jpg / jpeg / png)
4. Click **"Predict Demand"**
5. View predicted quantity, demand category, and sales context

---

## Tech Stack

| Tool | Purpose |
|---|---|
| EasyOCR | Extract product codes from image watermarks |
| OpenCV | Image preprocessing — bottom crop before OCR |
| MobileNetV2 | Visual feature extraction via transfer learning |
| GradientBoosting | Sales quantity regression model |
| Streamlit | Web UI for prediction |
| Pandas | Data cleaning, aggregation, and merging |
| Google Colab | Model training environment with free GPU |

---

## Scripts Overview

| File | What it does |
|---|---|
| `clean_csv.py` | Reads raw `sales_data.csv`, aggregates by product code, saves `aggregated_sales.csv` |
| `notebooks/pipeline.ipynb` | Full OCR extraction + feature extraction + model training on Colab |
| `app.py` | Streamlit UI — loads trained model, extracts features from uploaded image, shows prediction |

---
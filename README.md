# 🧠 Amazon Review Analysis — End-to-End NLP Pipeline

> *From raw messy text to business insights — a complete Natural Language Processing pipeline built on 50,000 real Amazon food reviews.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![spaCy](https://img.shields.io/badge/spaCy-3.x-09a3d5?style=flat-square)
![gensim](https://img.shields.io/badge/gensim-4.x-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Sentiment%20Accuracy-89%25-brightgreen?style=flat-square)
![Topics](https://img.shields.io/badge/LDA%20Topics-8-purple?style=flat-square)
![Reviews](https://img.shields.io/badge/Reviews%20Processed-50%2C000-red?style=flat-square)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Session 1 — Text Cleaning & Preprocessing](#-session-1--text-cleaning--preprocessing)
- [Session 2 — Bag of Words & TF-IDF](#-session-2--bag-of-words--tf-idf)
- [Session 3 — Sentiment Analysis](#-session-3--sentiment-analysis)
- [Session 4 — Topic Modeling with LDA](#-session-4--topic-modeling-with-lda)
- [Session 5 — Word Embeddings (Word2Vec)](#-session-5--word-embeddings-word2vec)
- [Session 6 — Named Entity Recognition](#-session-6--named-entity-recognition-ner)
- [Key Results & Insights](#-key-results--insights)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Project Structure](#-project-structure)
- [What I Learned](#-what-i-learned)
- [Next Steps](#-next-steps)

---

## 🎯 Project Overview

This project builds a **complete NLP intelligence pipeline** on the Amazon Fine Food Reviews dataset. Rather than isolated exercises, every NLP technique feeds into a single cohesive system that extracts real business insights from raw customer reviews.

**What this pipeline does:**
- Cleans and normalises raw messy review text
- Converts text into numerical representations (BoW, TF-IDF)
- Predicts whether a review is positive or negative with **89% accuracy**
- Discovers **8 hidden product themes** from reviews with zero labels
- Understands word meaning and relationships through embeddings
- Extracts brand names and entities automatically

**Why this matters:** A real business could plug any product reviews into this pipeline and instantly get sentiment scores, topic breakdowns, and brand mention tracking — without reading a single review manually.

---

## 📦 Dataset

**Source:** [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

| Property | Detail |
|---|---|
| Original size | ~568,000 reviews |
| Working sample | 50,000 reviews (random seed 42) |
| Columns used | `Score`, `Summary`, `Text` |
| Time period | Oct 1999 – Oct 2012 |
| Rating scale | 1–5 stars |

**Score Distribution (50k sample):**

```
⭐⭐⭐⭐⭐  5 stars  →  32,097  (64.2%)
⭐⭐⭐⭐    4 stars  →   7,008  (14.0%)
⭐⭐⭐      3 stars  →   3,791  ( 7.6%)
⭐⭐        2 stars  →   2,576  ( 5.2%)
⭐          1 star   →   4,528  ( 9.1%)
```

> ⚠️ **Class Imbalance Note:** The dataset is heavily skewed toward positive reviews (64% five-star). This was handled by converting to binary sentiment (positive: 4–5 stars, negative: 1–3 stars) and is a known limitation discussed in Session 3.

---

## 🏗️ Project Architecture

```
Raw Reviews (CSV)
        │
        ▼
┌─────────────────────┐
│  Session 1          │  → Lowercase, remove HTML/URLs/punctuation,
│  Text Cleaning      │    stopword removal, lemmatization
└────────┬────────────┘
         │  49.1% noise reduction
         ▼
┌─────────────────────┐
│  Session 2          │  → CountVectorizer (BoW) 50k×5000
│  BoW & TF-IDF       │    TfidfVectorizer 50k×5000
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────┐               ┌──────────────────────┐
│  Session 3      │               │  Session 4           │
│  Sentiment      │               │  Topic Modeling      │
│  Analysis       │               │  (LDA)               │
│  89% accuracy   │               │  8 topics discovered │
└─────────────────┘               └──────────────────────┘
         
┌─────────────────────┐
│  Session 5          │  → Word2Vec: 13,601 word vocabulary
│  Word Embeddings    │    100-dimensional vectors
└─────────────────────┘

┌─────────────────────┐
│  Session 6          │  → spaCy NER: brands, orgs, dates
│  NER                │    16 unique brands extracted
└─────────────────────┘
```

---

## 🧹 Session 1 — Text Cleaning & Preprocessing

### What we built
A reusable `clean_text()` function that transforms raw, noisy review text into clean tokens ready for ML.

### Pipeline steps
```
Raw Text
   │
   ├─ 1. Lowercase everything
   ├─ 2. Remove HTML tags  (<br>, <div>, etc.)
   ├─ 3. Remove URLs
   ├─ 4. Remove punctuation & numbers
   ├─ 5. Tokenize (split into words)
   ├─ 6. Remove stopwords (the, and, is, etc.)
   └─ 7. Lemmatize (running → run, cookies → cookie)
```

### Before vs After

```
BEFORE:
"Having tried a couple of other brands of gluten-free sandwich 
cookies, these are the best bunch I've tried so far!"

AFTER:
"tried couple brand glutenfree sandwich cooky best bunch"
```

### Key Results

| Metric | Value |
|---|---|
| Avg words BEFORE cleaning | 84.2 |
| Avg words AFTER cleaning | 42.8 |
| Vocabulary reduction | **49.1%** |
| Null values handled | ✅ (float NaN → empty string) |

### Key lesson
> Real-world text data almost always contains `NaN` values. Always add `fillna('')` and `isinstance(text, str)` checks before any string operations.

---

## 📊 Session 2 — Bag of Words & TF-IDF

### What we built
Two numerical representations of text, and a visualisation showing which words define each star rating.

### Bag of Words vs TF-IDF

| | Bag of Words | TF-IDF |
|---|---|---|
| Representation | Word counts | Weighted scores |
| Matrix shape | 50,000 × 5,000 | 50,000 × 5,000 |
| "coffee" across all reviews | High count | Low score (too common) |
| "glutenfree" in one review | Medium count | High score (distinctive) |
| Handles word importance | ❌ | ✅ |

### Signature words per star rating

```
⭐     1 star  →  product, taste, like, coffee, one, would, food, buy
⭐⭐    2 star  →  taste, like, coffee, flavor, product, would, good, tea
⭐⭐⭐   3 star  →  taste, like, good, coffee, flavor, product, would, tea
⭐⭐⭐⭐  4 star  →  good, coffee, like, taste, flavor, great, tea, product
⭐⭐⭐⭐⭐ 5 star  →  great, love, coffee, tea, good, like, product, best
```

> **Insight:** "would" appears in 1-star reviews — as in *"I would NOT buy again."* TF-IDF captures these subtle negative signals that BoW misses.

### WordCloud Output
Positive reviews dominated by: **love, good, delicious, best, wonderful**  
Negative reviews dominated by: **one, product, disappointed, make, even**

---

## 🤖 Session 3 — Sentiment Analysis

### What we built
A Logistic Regression classifier trained on TF-IDF vectors to predict positive/negative sentiment.

### Model configuration

```python
TfidfVectorizer(max_features=5000)
LogisticRegression(max_iter=1000, random_state=42)
train_test_split(test_size=0.2, stratify=y)
```

### Results

```
=== Classification Report ===

              precision    recall    f1-score   support
  Negative       0.82      0.63      0.71       2,179
  Positive       0.90      0.96      0.93       7,821

  accuracy                           0.89      10,000
```

### Confusion Matrix

```
                 Predicted
                 Neg    Pos
Actual  Neg  [ 1376    803 ]
        Pos  [  302   7519 ]
```

### Top predictive words

**→ Positive signals:**
`great (10.45)` · `best (8.36)` · `love (8.25)` · `delicious (8.10)` · `excellent (7.30)`

**→ Negative signals:**
`ok (-6.46)` · `disappointed (-6.32)` · `horrible (-5.67)` · `worst (-5.65)` · `unfortunately (-5.02)`

### Live predictions

```
"This coffee is absolutely amazing, I love it!"
→ POSITIVE 😊  Confidence: 99.9%

"Terrible product, complete waste of money, never buying again."
→ NEGATIVE 😞  Confidence: 99.4%

"It was okay, nothing special but not bad either."
→ NEGATIVE 😞  Confidence: 99.6%  ← "okay" scores -6.46!
```

### Why accuracy alone is misleading
With 64% positive reviews, a model predicting everything as positive gets 64% "accuracy" without learning anything. **F1-score per class** is the honest metric — our Negative F1 of 0.71 reveals real room for improvement that 89% accuracy hides.

---

## 🗂️ Session 4 — Topic Modeling with LDA

### What we built
An unsupervised LDA model that discovers hidden themes in reviews — with **zero labels**.

### Model configuration

```python
CountVectorizer(max_features=3000, min_df=5, max_df=0.90)
LatentDirichletAllocation(n_components=8, max_iter=15, learning_method='online')
```

### Discovered topics

| Topic | Top Words | Label |
|---|---|---|
| 1 | price, amazon, store, buy, find | 🛒 Shopping & Value |
| 2 | make, mix, sauce, add, use | 👨‍🍳 Cooking & Recipes |
| 3 | coffee, cup, flavor, strong, vanilla | ☕ Coffee |
| 4 | bag, box, package, review, got | 📦 Packaging & Delivery |
| 5 | tea, drink, flavor, water, chip | 🍵 Tea & Drinks |
| 6 | chocolate, snack, bar, cookie, delicious | 🍫 Snacks & Sweets |
| 7 | cat, ingredient, natural, diet, healthy | 🐱 Cat Food & Health |
| 8 | dog, treat, eat, love, old | 🐶 Dog Treats |

### Sentiment by topic

```
🛒 Shopping & Value      ████████████████████  4.57 ⭐
🍫 Snacks & Sweets       ███████████████████   4.52 ⭐
👨‍🍳 Cooking & Recipes    ██████████████████    4.40 ⭐
🐶 Dog Treats            ██████████████████    4.38 ⭐
☕ Coffee                 █████████████████     4.26 ⭐
🍵 Tea & Drinks          ████████████████      4.08 ⭐
🐱 Cat Food & Health     ███████████████       3.86 ⭐
📦 Packaging & Delivery  █████████████         3.31 ⭐  ← Most complaints
```

### Key business insight
> **Packaging & Delivery** scored lowest at 3.31 — customers are happy with the food but frustrated with how it arrives. This is an actionable insight a real Amazon seller would act on immediately.

---

## 🔤 Session 5 — Word Embeddings (Word2Vec)

### What we built
A Word2Vec model trained from scratch on our review corpus, converting every word into a 100-dimensional meaning vector.

### Model configuration

```python
Word2Vec(
    sentences=sentences,   # 50,000 tokenized reviews
    vector_size=100,       # each word = 100 numbers
    window=5,              # context window: 5 words left & right
    min_count=5,           # ignore rare words
    workers=4,
    epochs=10
)
# Vocabulary learned: 13,601 unique words
```

### Word similarity results

```
Similar to 'coffee':    expresso (0.769), starbucks (0.746), espresso (0.739), brew (0.704)
Similar to 'delicious': yummy (0.784), tasty (0.737), delish (0.714), wonderful (0.674)
Similar to 'dog':       puppy (0.837), pup (0.833), pug (0.762), chihuahua (0.762)
Similar to 'disappointed': disappointing (0.659), dissapointed (0.631), sorry (0.542)
```

### Word arithmetic (meaning math!)

```
good + strong - weak        =  great, decent, excellent ✅
coffee + sweet - bitter     =  cappuccino, eggnog, dessert ✅
dog + food - treat          =  purina, orijen, iams, canidae ✅ (actual brand names!)
```

> **The magic:** `dog + food - treat` returned real dog food brand names that the model never explicitly learned — it inferred them purely from word co-occurrence patterns.

### TF-IDF vs Word2Vec

| | TF-IDF | Word2Vec |
|---|---|---|
| Word representation | Single number (importance score) | 100 numbers (meaning vector) |
| "good" vs "great" | Completely different columns | Mathematically close vectors |
| Captures synonyms | ❌ | ✅ |
| Supports arithmetic | ❌ | ✅ |
| Analogy: | Price tag | GPS coordinates |

---

## 🏷️ Session 6 — Named Entity Recognition (NER)

### What we built
A spaCy NER pipeline that extracts brands, organisations, dates and quantities from raw review text — with zero training required.

### Model used
```python
spacy.load('en_core_web_sm')  # Pre-trained on news corpus
```

### Entity types extracted

| Label | Meaning | Example |
|---|---|---|
| ORG | Organisation/Brand | Starbucks, Walmart |
| PRODUCT | Product name | K-Cups |
| DATE | Time references | a few months |
| CARDINAL | Numbers/quantities | no more than 6 |
| PERSON | Person names | (sometimes misclassified brands) |

### Top brands extracted (1,000 review sample)

```
🏪 Amazon          133 mentions
☕ Starbucks         10 mentions
🫙 Senseo            7 mentions
🌿 Whole Foods       6 mentions
🌱 AeroGarden        6 mentions
🛒 Costco            5 mentions
⚖️ Weight Watchers   3 mentions
☕ Marley Coffee      3 mentions
```

### Real-world limitation
> spaCy was trained on **news articles**, not food reviews. This causes false positives — "LOVED", "Vanilla", "Coconut" get tagged as ORG. In production, a post-processing filter or domain-specific NER model would be needed.

---

## 📈 Key Results & Insights

### Model Performance Summary

| Component | Metric | Result |
|---|---|---|
| Text Preprocessing | Vocabulary reduction | **49.1%** |
| Sentiment Classifier | Overall accuracy | **89%** |
| Sentiment Classifier | Positive F1 | **0.93** |
| Sentiment Classifier | Negative F1 | **0.71** |
| Topic Model | Topics discovered | **8 meaningful themes** |
| Word2Vec | Vocabulary learned | **13,601 words** |
| NER | Unique brands found | **16 real brands** |

### Business Insights Generated

1. **Packaging is the #1 pain point** — 3.31 avg rating vs 4.57 for Shopping & Value
2. **"okay" is the most negative word** — coefficient -6.46, stronger signal than "horrible"
3. **Dog food brands learned by model** — Purina, Orijen, Iams inferred from context alone
4. **Tea & Drinks** has highest review volume (10,123) but mediocre sentiment (4.08) — competitive, unsatisfied category
5. **Lukewarm = Negative** — "It was okay, nothing special" predicted negative at 99.6% confidence

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.x | Data loading, manipulation |
| `numpy` | 1.x | Numerical operations |
| `nltk` | 3.x | Stopwords, lemmatization |
| `scikit-learn` | 1.x | TF-IDF, BoW, Logistic Regression, LDA |
| `gensim` | 4.x | Word2Vec embeddings |
| `spaCy` | 3.x | Named Entity Recognition |
| `textblob` | 0.x | Polarity scoring |
| `matplotlib` | 3.x | Visualisations |
| `seaborn` | 0.x | Confusion matrix heatmap |
| `wordcloud` | 1.x | WordCloud visualisations |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/amazon-review-nlp.git
cd amazon-review-nlp
```

### 2. Install dependencies
```bash
pip install pandas numpy nltk scikit-learn gensim spacy textblob matplotlib seaborn wordcloud
```

### 3. Download NLP models & corpora
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

```bash
python -m spacy download en_core_web_sm
```

### 4. Download the dataset
- Go to [Kaggle — Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Download `Reviews.csv`
- Place it in the project root folder

### 5. Run the notebooks in order
```
01_preprocessing.ipynb
02_tfidf_bow.ipynb
03_sentiment_analysis.ipynb
04_topic_modeling.ipynb
05_word2vec.ipynb
06_ner.ipynb
```

> **Tip:** Always run the master imports cell at the top of each notebook before anything else.

---

## 📁 Project Structure

```
amazon-review-nlp/
│
├── 📓 01_preprocessing.ipynb       # Text cleaning pipeline
├── 📓 02_tfidf_bow.ipynb           # BoW & TF-IDF matrices
├── 📓 03_sentiment_analysis.ipynb  # Sentiment classifier
├── 📓 04_topic_modeling.ipynb      # LDA topic model
├── 📓 05_word2vec.ipynb            # Word embeddings
├── 📓 06_ner.ipynb                 # Named entity recognition
│
├── 📄 utils.py                     # Shared clean_text() function
├── 📄 Reviews.csv                  # Raw dataset (download separately)
├── 📄 reviews_cleaned.csv          # Preprocessed data (generated)
│
├── 🖼️ wordclouds.png               # Positive vs negative wordclouds
├── 🖼️ confusion_matrix.png         # Sentiment model evaluation
├── 🖼️ topic_sentiment.png          # Avg rating by topic
├── 🖼️ word2vec_clusters.png        # 2D word embedding visualisation
│
└── 📄 README.md
```

---

## 💡 What I Learned

### Technical lessons
- **NaN handling is non-negotiable** — real text data always has missing values; `fillna('')` and `isinstance(text, str)` guards are essential
- **Accuracy is a misleading metric** for imbalanced datasets — always report F1 per class
- **TF-IDF > BoW** for sentiment because it down-weights generic words like "coffee" that appear across all ratings
- **Word2Vec learns syntax AND semantics** — "disappointed", "impressed", "pleased" cluster together because they share grammatical context, not just meaning
- **LDA is purely statistical** — it groups words by co-occurrence, not meaning. The fact that it produces interpretable topics is an emergent property of language itself
- **Domain mismatch matters** — spaCy trained on news misclassifies food review entities; production NER needs domain-specific training

### Conceptual takeaways

| Concept | One-liner |
|---|---|
| Lemmatization | Makes "running", "ran", "runs" the same token |
| Stopwords | "the", "and", "is" — frequent but meaningless |
| TF-IDF | Rewards words distinctive to a document, penalises generic ones |
| LDA | Groups words that travel together across documents |
| Word2Vec | "Tell me your neighbours and I'll tell you who you are" |
| NER | Finds real-world named things without knowing what they are |

---

## 🚀 Next Steps

### Intermediate improvements
- [ ] **Fix class imbalance** — use `class_weight='balanced'` in Logistic Regression to improve Negative F1 from 0.71
- [ ] **Try VADER** instead of TextBlob — purpose-built for social/review text
- [ ] **Compare classifiers** — Random Forest, XGBoost vs Logistic Regression on TF-IDF
- [ ] **Add bigrams** to TF-IDF — "not good" means something very different from "good"

### Advanced extensions
- [ ] **GloVe pretrained vectors** — trained on 840B tokens vs our 50k reviews
- [ ] **HuggingFace BERT** — contextual embeddings, expected accuracy 93–95%
- [ ] **Streamlit app** — deploy the pipeline so anyone can paste a review and get predictions
- [ ] **Domain-specific NER** — fine-tune spaCy on food review entities

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built as a hands-on NLP learning project covering the full classical NLP pipeline from preprocessing to deployment-ready insights.

*If you found this useful, give it a ⭐ on GitHub!*

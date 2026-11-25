# Explainable Fake Review Detection System using Machine Learning and SHAP Analysis

**Student ID:** c5044513 · MSc Dissertation, Sheffield Hallam University

This project builds an explainable machine learning system that sorts a hotel review into one of three classes:

- **Genuine**: real customer reviews (TripAdvisor and Web)
- **Human Fake**: deceptive reviews written by people (MTurk)
- **AI Generated**: reviews written by an AI model

Four classifiers are compared on TF-IDF text features. XGBoost is tuned and then explained with **SHAP** to show which words drive each prediction. The models are also checked on a different domain (Amazon product reviews).

---

## Repository contents

| File | Description |
|---|---|
| `C5044513_dissertation_report 1.docx` | Final dissertation report |
| `C5044513_dissertation_PPT (1).pdf` | Presentation slides |
| `c5044513_Practical_Final.ipynb` | Main code notebook (practical implementation, with outputs) |
| `c5044513_Practical_Final.html` | HTML copy of the practical notebook, viewable in any browser |
| `c5044513_Overview_Final.ipynb` | Code notebook with the full written overview (data sources, variables, method and conclusions) |
| `c5044513_Overview_Final.html` | HTML copy of the overview notebook |
| `c5044513_Website.html` | Interactive web demo: paste a review and see the verdict and why it was flagged |
| `deceptive-opinion-augmented.csv` | Main dataset: 2,400 hotel reviews (800 per class) |
| `amazon_genuine_sample.csv` | 2,000 Amazon genuine product reviews, used only for the generalisation check |

---

## Datasets

| Class | Reviews | Source |
|---|---:|---|
| Genuine (`truthful`) | 800 | TripAdvisor (400) and Web (400) |
| Human Fake (`deceptive`) | 800 | MTurk crowd-workers |
| AI Generated (`AI generated`) | 800 | Generated with an AI model in the style of the original corpus |

- The Genuine and Human Fake reviews come from the **Deceptive Opinion Spam Corpus** (Chicago hotel reviews). The AI Generated class was added to make a balanced 3-class dataset.
- Polarity is balanced (1,200 positive and 1,200 negative).
- Only the review text is used as model input. The `source` column was excluded to avoid data leakage.
- After cleaning, **1,788 Amazon genuine reviews** remain. A sample of 400 is used to test generalisation; none are used for training.

---

## Method

1. **Preprocessing:** lowercasing, removing non-letters, stop-word removal, lemmatisation (NLTK WordNet)
2. **Features:** TF-IDF with 6,000 unigrams and bigrams (`min_df=3`, `max_df=0.90`), fitted on the training set only
3. **Split:** stratified 80/20, giving 1,920 training and 480 test reviews
4. **Models:** Logistic Regression, Naive Bayes, Random Forest and XGBoost, all with default settings
5. **Tuning:** XGBoost with `RandomizedSearchCV` (12 settings, 3-fold CV, macro F1)
6. **Generalisation:** the tuned XGBoost and Logistic Regression predict 400 Amazon genuine reviews
7. **Explainability:** SHAP `TreeExplainer` on 200 test reviews, with top words for each class

---

## Results

**Hotel test set (480 reviews)**

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | **93.33%** | 0.9330 |
| Naive Bayes | 92.50% | 0.9248 |
| XGBoost (default) | 88.54% | 0.8848 |
| Tuned XGBoost | 87.50% | 0.8747 |
| Random Forest | 83.75% | 0.8339 |

- **Best tuned XGBoost settings:** `n_estimators=300`, `max_depth=3`, `learning_rate=0.1`, `min_child_weight=3`, `gamma=0.1`, `subsample=1.0`, `colsample_bytree=1.0`. Best cross-validation macro F1 was 0.8548.
- The tuned XGBoost classified **420 of 480** test reviews correctly. AI Generated reviews were the easiest class (155 of 160). Most errors (47 of 60) were between Genuine and Human Fake.

**Generalisation on 400 Amazon genuine reviews**

| Model | Predicted Genuine | Human Fake | AI Generated |
|---|---:|---:|---:|
| Tuned XGBoost | **368 (92.0%)** | 32 | 0 |
| Logistic Regression | 330 (82.5%) | 57 | 13 |

Logistic Regression scored highest on the hotel data, but the tuned XGBoost generalised better to a new domain.

**SHAP: top words for each class**

- **Genuine:** place names and specific stay details (*chicago, location, michigan, arrived, smelled, construction*)
- **Human Fake:** room and hotel features plus emphatic words (*small, floor, quiet, breakfast, elevator, definitely, luxury*)
- **AI Generated:** generic hotel-service and narration words (*hotel, room, bed, staff, desk, said, asked, front desk*)

---

## How to run the notebooks

**Requirements:** Python 3.10 or later, plus Jupyter (VS Code, JupyterLab or Jupyter Notebook)

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost shap ipykernel
```

1. Download or clone this repository.
2. Keep both `.csv` files in the **same folder** as the notebooks.
3. Open `c5044513_Practical_Final.ipynb` (or the Overview notebook) and choose **Run All**.
   - The first cell downloads the NLTK resources it needs (stopwords, punkt, wordnet), so an internet connection is needed the first time.
   - A full run takes about 3 to 5 minutes. Most of that is the XGBoost tuning step.

Results can differ slightly with other library versions (for example XGBoost or scikit-learn). The figures above come from the submitted run, which is saved in the notebooks and HTML files.

To view the results without running anything, open the `.html` files in a web browser.

---

## Web demo

Open `c5044513_Website.html` in any web browser. Nothing needs to be installed.

1. The home page shows the project title. Click **"Click here to check a review"**.
2. Paste a hotel review, or use one of the three example buttons, then press **Check review**.
3. The page shows:
   - the verdict (Genuine, Human Fake or AI Generated) with a confidence bar for each class
   - **Why it was flagged:** key points, including the clue words and the typical pattern for that class
   - **Words behind the decision:** the review with its important words highlighted by class
   - a list of all reviews checked

The demo runs a Logistic Regression model inside the browser. It uses the same preprocessing, TF-IDF settings and 80/20 split as the notebooks and gets 93.75% accuracy on the test set. Its lemmatiser is simplified, which explains the small difference from the notebook's 93.33%. It was trained on Chicago hotel reviews, so other kinds of text give less reliable results.

---

## Limitations

- The AI reviews come from one tool and one prompt, so the results may not hold for other AI writers.
- The Amazon sample contains genuine reviews only, so it measures false flags rather than full detection accuracy.
- Only XGBoost was tuned, and the results come from a single 80/20 split.
- Some strong SHAP words (*chicago, michigan*) are specific to the hotel corpus and may not transfer to other domains.
- RQ4 (whether SHAP explanations increase user confidence) is outlined for a human evaluation and is not tested in the notebooks.

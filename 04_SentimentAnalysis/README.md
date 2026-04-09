# Discrete emotion classification (German news titles)

Sentence-level inference with a fine-tuned **GELECTRA** model for **8 discrete emotions** in German political text, following Widmann & Wich (2023), *Political Analysis*.

**Emotions:** anger, fear, disgust, sadness, joy, enthusiasm, pride, hope.

---

## Step-by-step: test the pipeline (beginner-friendly)

Do these in order. Replace `/Users/gretawette/Documents/Thesis` with your own project folder if it differs.

### Step 1 — Open a terminal

- **Mac:** open **Terminal** (Spotlight → type “Terminal”).
- You will type commands and press **Enter** after each line.

### Step 2 — Go to your Thesis project folder

```bash
cd /Users/gretawette/Documents/Thesis
```

### Step 3 — Use your Python environment (recommended)

If you use the project’s virtual environment:

```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of the line.

### Step 4 — Install Python packages (once)

```bash
pip install -r Sentiment_Analysis/requirements.txt
```

Wait until it finishes without errors.

### Step 5 — Download the fine-tuned model weights (~445 MB)

`config.json` is already in the repo. You only need the large **`pytorch_model.bin`** file:

```bash
python Sentiment_Analysis/download_model_weights.py
```

- This needs **internet** and may take **several minutes**.
- When it says **Done** and shows a size around **425 MB**, you’re good.

**Manual option:** download [pytorch_model.bin](https://github.com/tweedmann/3x8emotions/releases/download/electra-model/pytorch_model.bin) from the [electra-model release](https://github.com/tweedmann/3x8emotions/releases/tag/electra-model) and move it into:

`Sentiment_Analysis/models/final/german-nlp-group/electra-base-german-uncased/`

### Step 6 — Run the small test (about 100 titles)

```bash
cd Sentiment_Analysis
python emotion_sample_test.py
```

- The first run may also download tokenizer files from Hugging Face (normal).
- At the end you should see **Results saved to** `outputs/emotion_sample_results.csv`.

### Step 7 — (Later) Run on the full dataset

Only after Step 6 looks good:

```bash
python emotion_pipeline.py
```

This processes **all** cleaned titles in `00_Initial EDA/df_combined.csv` and can take a long time on CPU.

---

## Difference from `3a_SentimentAnalysis/.../02_run_gelectra.py`

- **Inspiration code** applies the model to many sentences per press release (sentence split → aggregate).
- **Here**, each row’s **`Title`** is treated as **one sentence and one document** — no splitting or aggregation.

## Model files (what lives where)

| File | Source |
|------|--------|
| `config.json` | **Included** in this repo under `models/final/.../electra-base-german-uncased/` |
| `pytorch_model.bin` | **Download** with `download_model_weights.py` or manually from [3x8emotions release “electra-model”](https://github.com/tweedmann/3x8emotions/releases/tag/electra-model) |
| Tokenizer | **Auto-downloaded** from Hugging Face (`german-nlp-group/electra-base-german-uncased`) the first time you run inference |

## Label order

Output columns follow the order used in the official `helper/inferencing.py`:  
`anger, fear, disgust, sadness, joy, enthusiasm, pride, hope` (indices 0–7).  
The released `config.json` lists `LABEL_0`…`LABEL_7`; we map them to these names.

## Probability outputs

The original repo uses **sigmoid** + threshold (multi-label). This pipeline applies **softmax** over logits so the eight scores **sum to ~1** per title and `emotion_dominant` is the **argmax**. That supports descriptive “distribution over emotions” and dominant-label summaries; it is not identical to the paper’s multi-label decision rule.

## Quick command recap

```bash
cd /path/to/Thesis
source .venv/bin/activate   # if you use the project venv
pip install -r Sentiment_Analysis/requirements.txt
python Sentiment_Analysis/download_model_weights.py
cd Sentiment_Analysis
python emotion_sample_test.py    # test first
python emotion_pipeline.py       # full corpus when ready
```

Optional environment variables:

- `EMOTION_DATA_PATH` — default: `../00_Initial EDA/df_combined.csv`
- `EMOTION_MODEL_DIR` — default: `models/final/german-nlp-group/electra-base-german-uncased`
- `EMOTION_OUTPUT_PATH` — full run output CSV (pipeline only)

## Outputs

| File | Produced by |
|------|-------------|
| `outputs/emotion_sample_results.csv` | `emotion_sample_test.py` |
| `outputs/emotion_full_results.csv` | `emotion_pipeline.py` (default) |
| `outputs/figures/*.png` | `04_visualize_emotions.ipynb` (after full run) |
| `outputs/tables_*.csv` | same notebook (summary exports) |

### Visualization notebook

After `emotion_full_results.csv` exists, open **`04_visualize_emotions.ipynb`** (run from the `Sentiment_Analysis` folder). It mirrors the structure of `3a_SentimentAnalysis/.../04_visualize.ipynb`: setup → summary tables → seaborn/matplotlib plots (corpus means, heatmap by outlet, dominant-emotion shares, boxplots, monthly trends). Install `matplotlib` and `seaborn` via `requirements.txt`.

## Citation

Widmann, T., & Wich, M. (2023). Creating and Comparing Dictionary, Word Embedding, and Transformer-Based Models to Measure Discrete Emotions in German Political Text. *Political Analysis*, 31(4), 626–641.

Use of the pretrained weights should also reference the **3x8emotions** repository as indicated there.

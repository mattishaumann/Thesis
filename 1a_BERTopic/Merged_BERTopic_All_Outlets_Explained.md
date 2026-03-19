# Merged_BERTopic_All_Outlets.ipynb Explained

This document explains what [Merged_BERTopic_All_Outlets.ipynb](/Users/MattisHaumann/Dev/Thesis/1a_BERTopic/Merged_BERTopic_All_Outlets.ipynb) does.

It is meant as a companion file you can hand to another LLM.

Important context:
- The notebook is not fully self-contained.
- It imports helper logic from [merged_outlets_analysis.py](/Users/MattisHaumann/Dev/Thesis/1a_BERTopic/merged_outlets_analysis.py).
- So this explanation covers:
  - the notebook cell-by-cell
  - the meaning of each line of code in the notebook
  - the role of the imported helper functions and constants

## High-Level Purpose

The notebook does six things:

1. Finds the repo root and the saved BERTopic model folders.
2. Loads seven already-trained BERTopic models:
   - Tagesschau
   - RT
   - Antispiegel
   - Tichys Einblick
   - Nius
   - Compact
   - Deutschlandkurier
3. Merges those outlet-specific topic models into one shared merged model with `BERTopic.merge_models(...)`.
4. Rebuilds the prepared article corpora for all outlets, so the merged model can be applied to article-level documents.
5. Projects the merged article embeddings into 2D with UMAP and plots the shared topic space.
6. Produces outlet-level comparison plots where:
   - all documents are grey
   - one alternative-media outlet at a time is red
   - topic labels are shown for the topics that outlet covers
   - a summary table reports how many merged topics each alternative-media outlet covers

## Execution Order

The notebook is meant to run top to bottom:

1. Cell 1: path setup
2. Cell 2: imports and model loading
3. Cell 3: model merge
4. Cell 4: optional merged-model save
5. Cell 5: rebuild prepared outlet documents
6. Cell 6: transform all prepared articles into merged-topic assignments and UMAP coordinates
7. Cell 7: global merged-topic UMAP
8. Cell 8: outlet coverage summary plus one red-vs-grey plot per alternative-media outlet

## Notebook Cells

### Cell 0

```markdown
# Merge BERTopic Models Across Outlets

This notebook loads the saved BERTopic models for Tagesschau, RT, Antispiegel, Tichys Einblick, Nius, Compact, and Deutschlandkurier, merges them with `BERTopic.merge_models(...)`, then builds merged article-level UMAP maps.
```

What it does:
- States the notebook purpose.
- Tells the reader that this is a model-merging notebook, not an outlet-training notebook.
- Signals that the final output is article-level UMAP visualization in the merged topic space.

### Cell 1

```python
import os
import sys
from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SIMILARITY = 0.7


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing .git")


PROJECT_ROOT = find_project_root(Path.cwd())
MODULE_ROOT = PROJECT_ROOT / "1a_BERTopic"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

os.environ["NUMBA_CACHE_DIR"] = str(PROJECT_ROOT / ".numba_cache")

MODEL_DIR_CANDIDATES = [
    PROJECT_ROOT / "1a_BERTopic" / "local_outputs",
    PROJECT_ROOT / "1a_BERTopic" / "outputs",
    PROJECT_ROOT / "BERTopic" / "outputs",
]
MERGED_SAVE_DIR = PROJECT_ROOT / "1a_BERTopic" / "local_outputs" / "merged_all_outlets_model"

print(f"Project root: {PROJECT_ROOT}")
print("Model path candidates:")
for candidate in MODEL_DIR_CANDIDATES:
    print(f"  - {candidate}")
```

Line-by-line:
- `import os`
  - Needed so the notebook can set the `NUMBA_CACHE_DIR` environment variable.
- `import sys`
  - Needed so the notebook can modify Python's import path.
- `from pathlib import Path`
  - Used for safe filesystem path construction.
- `EMBEDDING_MODEL = ...`
  - Defines the sentence-transformer used when loading saved BERTopic models and when merging them.
- `MIN_SIMILARITY = 0.7`
  - Sets the threshold used by `BERTopic.merge_models(...)` to decide whether topics from different outlet models are similar enough to merge.
- `def find_project_root(start: Path) -> Path:`
  - Defines a helper function that finds the repository root by looking for `.git`.
- `start = start.resolve()`
  - Converts the incoming path to an absolute resolved path.
- `for candidate in (start, *start.parents):`
  - Checks the current directory and then walks upward through parent directories.
- `if (candidate / ".git").exists():`
  - Tests whether that directory is the git repo root.
- `return candidate`
  - Returns the first directory that contains `.git`.
- `raise FileNotFoundError(...)`
  - Fails early if the notebook is run outside the repo.
- `PROJECT_ROOT = find_project_root(Path.cwd())`
  - Finds the repo root from the current working directory.
- `MODULE_ROOT = PROJECT_ROOT / "1a_BERTopic"`
  - Builds the path to the local BERTopic helper-code folder.
- `if str(MODULE_ROOT) not in sys.path:`
  - Checks whether Python already knows where to import helper modules from.
- `sys.path.insert(0, str(MODULE_ROOT))`
  - Adds the helper-code folder to the front of the import path.
- `os.environ["NUMBA_CACHE_DIR"] = ...`
  - Forces Numba cache files into a writable repo-local directory, avoiding cache issues in notebooks.
- `MODEL_DIR_CANDIDATES = [...]`
  - Defines the search order for saved BERTopic models.
  - `local_outputs` comes first so local reruns override tracked artifacts.
  - `outputs` comes second as the tracked fallback.
  - `BERTopic/outputs` is a legacy fallback from the older repo layout.
- `MERGED_SAVE_DIR = ...`
  - Defines where the optional merged model will be saved if saving is enabled later.
- `print(f"Project root: {PROJECT_ROOT}")`
  - Prints the resolved repo root for debugging.
- `print("Model path candidates:")`
  - Prints a header before listing search paths.
- `for candidate in MODEL_DIR_CANDIDATES:`
  - Iterates over candidate model directories.
- `print(f"  - {candidate}")`
  - Prints each candidate directory so the user can see where the notebook will search.

### Cell 2

```python
import importlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from bertopic import BERTopic

import merged_outlets_analysis as moa
moa = importlib.reload(moa)

ALT_MEDIA_OUTLET_KEYS = getattr(
    moa,
    "ALT_MEDIA_OUTLET_KEYS",
    ("rt", "antispiegel", "tichys", "nius", "compact", "deutschlandkurier"),
)
OUTLET_SPECS = moa.OUTLET_SPECS
build_merged_article_frame = moa.build_merged_article_frame
build_outlet_topic_coverage_summary = moa.build_outlet_topic_coverage_summary
combine_prepared_documents = moa.combine_prepared_documents
load_all_prepared_documents = moa.load_all_prepared_documents
plot_merged_topic_umap = moa.plot_merged_topic_umap
plot_outlet_highlight_umap = moa.plot_outlet_highlight_umap
resolve_model_paths = moa.resolve_model_paths

MODEL_PATHS = resolve_model_paths(MODEL_DIR_CANDIDATES)

loaded_models = {
    key: BERTopic.load(model_path, embedding_model=EMBEDDING_MODEL)
    for key, model_path in MODEL_PATHS.items()
}

for key, model in loaded_models.items():
    topic_info = model.get_topic_info()
    print(f"Loaded {key}: {MODEL_PATHS[key]} ({len(topic_info)} rows in topic info)")

tm_ts = loaded_models["tagesschau"]
tm_rt = loaded_models["rt"]
tm_as = loaded_models["antispiegel"]
tm_te = loaded_models["tichys"]
tm_ns = loaded_models["nius"]
tm_cm = loaded_models["compact"]
tm_dk = loaded_models["deutschlandkurier"]
```

Line-by-line:
- `import importlib`
  - Used to reload the helper module so stale Jupyter imports do not break the notebook.
- `import matplotlib.pyplot as plt`
  - Needed for plotting UMAP figures.
- `import pandas as pd`
  - Needed for table creation and display.
- `from IPython.display import display`
  - Used to display DataFrames in notebook cells.
- `from bertopic import BERTopic`
  - Imports the BERTopic class itself.
- `import merged_outlets_analysis as moa`
  - Imports the helper module that contains outlet-specific loaders, configs, and plot functions.
- `moa = importlib.reload(moa)`
  - Forces a reload of that helper module so notebook restarts are not required after edits.
- `ALT_MEDIA_OUTLET_KEYS = getattr(...)`
  - Reads the tuple of alternative-media outlets from the helper module.
  - If the helper constant is missing, it falls back to a hardcoded tuple.
- `OUTLET_SPECS = moa.OUTLET_SPECS`
  - Pulls the outlet metadata/config mapping into the notebook namespace.
- `build_merged_article_frame = moa.build_merged_article_frame`
  - Imports the helper that applies the merged model to all prepared documents and computes 2D coordinates.
- `build_outlet_topic_coverage_summary = moa.build_outlet_topic_coverage_summary`
  - Imports the helper that summarizes how many merged topics each outlet covers.
- `combine_prepared_documents = moa.combine_prepared_documents`
  - Imports the helper that concatenates all prepared outlet corpora.
- `load_all_prepared_documents = moa.load_all_prepared_documents`
  - Imports the helper that rebuilds prepared documents for all outlets.
- `plot_merged_topic_umap = moa.plot_merged_topic_umap`
  - Imports the helper for the global merged-topic map.
- `plot_outlet_highlight_umap = moa.plot_outlet_highlight_umap`
  - Imports the helper for one-outlet-red, all-others-grey plots.
- `resolve_model_paths = moa.resolve_model_paths`
  - Imports the helper that finds the actual saved model directories.
- `MODEL_PATHS = resolve_model_paths(MODEL_DIR_CANDIDATES)`
  - Resolves the location of each outlet model by searching the candidate directories from Cell 1.
- `loaded_models = { ... }`
  - Loads every BERTopic model from disk into memory.
- `key: BERTopic.load(model_path, embedding_model=EMBEDDING_MODEL)`
  - Loads one model at a time and tells BERTopic which embedding model to attach.
- `for key, model in loaded_models.items():`
  - Iterates over all loaded models.
- `topic_info = model.get_topic_info()`
  - Fetches the topic table for each loaded model.
- `print(f"Loaded {key}: ...")`
  - Prints the outlet key, actual path, and topic-table length as a quick sanity check.
- `tm_ts = loaded_models["tagesschau"]`
  - Creates a short variable for the Tagesschau model.
- `tm_rt = loaded_models["rt"]`
  - Creates a short variable for the RT model.
- `tm_as = loaded_models["antispiegel"]`
  - Creates a short variable for the Antispiegel model.
- `tm_te = loaded_models["tichys"]`
  - Creates a short variable for the Tichys Einblick model.
- `tm_ns = loaded_models["nius"]`
  - Creates a short variable for the Nius model.
- `tm_cm = loaded_models["compact"]`
  - Creates a short variable for the Compact model.
- `tm_dk = loaded_models["deutschlandkurier"]`
  - Creates a short variable for the Deutschlandkurier model.

### Cell 3

```python
models_to_merge = [
    tm_ts,
    tm_rt,
    tm_as,
    tm_te,
    tm_ns,
    tm_cm,
    tm_dk,
]

merged_model = BERTopic.merge_models(
    models_to_merge,
    min_similarity=MIN_SIMILARITY,
    embedding_model=EMBEDDING_MODEL,
)

merged_topic_info = merged_model.get_topic_info()
display(merged_topic_info.head(30))
print("Merged topic count:", len(merged_topic_info))
```

Line-by-line:
- `models_to_merge = [ ... ]`
  - Builds an ordered list of the seven outlet models to be merged.
- `tm_ts, tm_rt, tm_as, tm_te, tm_ns, tm_cm, tm_dk`
  - Explicitly defines which loaded models will participate in the merge.
- `merged_model = BERTopic.merge_models(...)`
  - Creates one merged BERTopic model from the seven outlet models.
- `models_to_merge`
  - Passes the list of model objects.
- `min_similarity=MIN_SIMILARITY`
  - Uses the threshold from Cell 1 to determine whether topics across models are similar enough to merge.
- `embedding_model=EMBEDDING_MODEL`
  - Ensures the merge process uses the same embedding model family.
- `merged_topic_info = merged_model.get_topic_info()`
  - Gets the merged model's topic summary table.
- `display(merged_topic_info.head(30))`
  - Shows the first 30 rows of merged topic info.
- `print("Merged topic count:", len(merged_topic_info))`
  - Prints the total number of rows in the merged topic table.

### Cell 4

```python
import shutil

SAVE_MERGED_MODEL = False

if SAVE_MERGED_MODEL:
    MERGED_SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MERGED_SAVE_DIR.exists():
        shutil.rmtree(MERGED_SAVE_DIR)
    merged_model.save(
        MERGED_SAVE_DIR,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL,
    )
    print(f"Saved merged model to: {MERGED_SAVE_DIR}")
else:
    print("Skipped save. Set SAVE_MERGED_MODEL = True to persist the merged model.")
```

Line-by-line:
- `import shutil`
  - Needed so an existing save directory can be removed before writing a fresh merged model.
- `SAVE_MERGED_MODEL = False`
  - Safety switch. By default, the notebook does not write a merged model to disk.
- `if SAVE_MERGED_MODEL:`
  - Only enters the save block if the user changes the flag to `True`.
- `MERGED_SAVE_DIR.parent.mkdir(parents=True, exist_ok=True)`
  - Creates the parent directory if it does not exist yet.
- `if MERGED_SAVE_DIR.exists():`
  - Checks whether an older merged model directory already exists.
- `shutil.rmtree(MERGED_SAVE_DIR)`
  - Deletes the old merged model directory so the new save is clean.
- `merged_model.save(...)`
  - Saves the merged BERTopic model.
- `MERGED_SAVE_DIR`
  - Uses the save path defined in Cell 1.
- `serialization="safetensors"`
  - Saves in the safer `safetensors` format.
- `save_ctfidf=True`
  - Saves the c-TF-IDF representation files too.
- `save_embedding_model=EMBEDDING_MODEL`
  - Stores the embedding model reference with the saved model.
- `print(f"Saved merged model to: {MERGED_SAVE_DIR}")`
  - Confirms the save location.
- `else:`
  - If saving is disabled, the notebook takes no write action.
- `print("Skipped save...")`
  - Explicitly tells the user that saving was skipped.

### Cell 5

```python
import pandas as pd

prepared_by_outlet = load_all_prepared_documents(PROJECT_ROOT)
prepared_summary = pd.DataFrame(
    [
        {
            "Outlet": OUTLET_SPECS[key].label,
            "Prepared_Documents": len(df),
        }
        for key, df in prepared_by_outlet.items()
    ]
).sort_values("Outlet").reset_index(drop=True)
display(prepared_summary)

combined_prepared = combine_prepared_documents(prepared_by_outlet)
print("Combined prepared documents:", len(combined_prepared))
```

Line-by-line:
- `import pandas as pd`
  - Re-imports pandas in this cell so the cell works even if run after a partial restart or out of order.
- `prepared_by_outlet = load_all_prepared_documents(PROJECT_ROOT)`
  - Rebuilds the prepared article tables for every outlet using the helper module.
- `prepared_summary = pd.DataFrame(...)`
  - Builds a summary DataFrame describing how many prepared documents each outlet contributes.
- `for key, df in prepared_by_outlet.items()`
  - Iterates over the outlet-to-DataFrame mapping returned by the helper.
- `"Outlet": OUTLET_SPECS[key].label`
  - Converts the internal outlet key into a user-facing outlet label.
- `"Prepared_Documents": len(df)`
  - Counts how many prepared documents that outlet contributes.
- `.sort_values("Outlet")`
  - Sorts the summary alphabetically by outlet name.
- `.reset_index(drop=True)`
  - Resets the DataFrame index so the display is clean.
- `display(prepared_summary)`
  - Shows the outlet preparation summary.
- `combined_prepared = combine_prepared_documents(prepared_by_outlet)`
  - Concatenates all prepared outlet DataFrames into one combined article table.
- `print("Combined prepared documents:", len(combined_prepared))`
  - Prints the total number of prepared documents across all outlets.

### Cell 6

```python
merged_articles, merged_topic_info_display, merged_umap_model = build_merged_article_frame(
    merged_model,
    combined_prepared,
)

display(merged_articles[["outlet_label", "document_id", "merged_topic", "merged_display_label"]].head())
display(merged_articles.groupby("outlet_label").size().rename("Article_Count").reset_index())
```

Line-by-line:
- `merged_articles, merged_topic_info_display, merged_umap_model = build_merged_article_frame(...)`
  - Calls the helper that:
    - applies the merged model to all combined prepared documents
    - gets article-level merged-topic assignments
    - extracts embeddings
    - reduces them to 2D with UMAP
    - returns a display-friendly merged topic table
- `merged_model`
  - Supplies the merged BERTopic model from Cell 3.
- `combined_prepared`
  - Supplies the prepared article table from Cell 5.
- `display(merged_articles[[...]].head())`
  - Displays a preview of the article-level merged result.
  - The preview includes:
    - outlet label
    - document id
    - merged topic id
    - display label for the merged topic
- `display(merged_articles.groupby("outlet_label").size().rename("Article_Count").reset_index())`
  - Shows how many merged articles each outlet contributes to the shared space.

### Cell 7

```python
fig, ax = plot_merged_topic_umap(
    merged_articles,
    merged_topic_info_display,
    top_n=20,
)
plt.show()
```

Line-by-line:
- `fig, ax = plot_merged_topic_umap(...)`
  - Calls the helper that creates the global merged-topic UMAP figure.
- `merged_articles`
  - Supplies the article-level 2D coordinates and topic assignments.
- `merged_topic_info_display`
  - Supplies the display-friendly topic labels.
- `top_n=20`
  - Requests labels/colors for the top 20 merged topics.
- `plt.show()`
  - Renders the plot in the notebook.

### Cell 8

```python
coverage_summary = build_outlet_topic_coverage_summary(
    merged_articles,
    merged_topic_info_display,
)
display(coverage_summary)

for outlet_key in ALT_MEDIA_OUTLET_KEYS:
    spec = OUTLET_SPECS[outlet_key]
    print(f"Plotting alternative-media outlet coverage: {spec.label}")
    fig, ax = plot_outlet_highlight_umap(
        merged_articles,
        outlet_key,
        merged_topic_info=merged_topic_info_display,
        top_n_labels=12,
    )
    plt.show()
```

Line-by-line:
- `coverage_summary = build_outlet_topic_coverage_summary(...)`
  - Calls the helper that computes how many merged topics each alternative-media outlet covers.
- `merged_articles`
  - Supplies article-level outlet and merged-topic assignments.
- `merged_topic_info_display`
  - Supplies the merged topic table so the helper knows how many non-outlier merged topics exist overall.
- `display(coverage_summary)`
  - Shows the summary table.
- `for outlet_key in ALT_MEDIA_OUTLET_KEYS:`
  - Loops through the alternative-media outlets only.
- `spec = OUTLET_SPECS[outlet_key]`
  - Fetches the outlet metadata for the current outlet key.
- `print(f"Plotting alternative-media outlet coverage: {spec.label}")`
  - Prints which outlet is about to be plotted.
- `fig, ax = plot_outlet_highlight_umap(...)`
  - Creates the red-vs-grey outlet plot for the current outlet.
- `merged_articles`
  - Supplies the shared 2D article coordinates and merged-topic assignments.
- `outlet_key`
  - Tells the helper which outlet should be highlighted in red.
- `merged_topic_info=merged_topic_info_display`
  - Supplies the merged topic table so the helper can label covered topics.
- `top_n_labels=12`
  - Limits the number of topic labels placed on each outlet map.
- `plt.show()`
  - Renders that outlet's plot.

## What The Imported Helper Module Does

The notebook depends on [merged_outlets_analysis.py](/Users/MattisHaumann/Dev/Thesis/1a_BERTopic/merged_outlets_analysis.py). These are the imported objects and what they mean.

### `OUTLET_SPECS`

This is the central outlet metadata dictionary.

For each outlet it stores:
- outlet key
- display label
- expected saved-model folder name
- text column to model
- id column
- outlet-specific BERTopic config factory
- outlet-specific raw-data loader

This matters because the merged notebook does not just load models. It also rebuilds the prepared article corpora consistently per outlet.

### `ALT_MEDIA_OUTLET_KEYS`

This tuple lists only the alternative-media outlets:
- `rt`
- `antispiegel`
- `tichys`
- `nius`
- `compact`
- `deutschlandkurier`

The notebook uses this tuple in Cell 8 so that the red-vs-grey loop excludes Tagesschau.

### `resolve_model_paths`

This helper:
- searches `local_outputs`, then `outputs`, then legacy `BERTopic/outputs`
- returns a dictionary of resolved model paths

The notebook uses it in Cell 2 before loading BERTopic models.

### `load_all_prepared_documents`

This helper:
- rebuilds a prepared document DataFrame for every outlet
- applies outlet-specific cleaning and filtering
- applies outlet-specific BERTopic preprocessing settings through `prepare_documents(...)`

The notebook uses it in Cell 5.

### `combine_prepared_documents`

This helper:
- concatenates the prepared document tables from all outlets
- preserves a consistent outlet ordering

The notebook uses it in Cell 5.

### `build_merged_article_frame`

This helper:
- transforms all combined prepared documents with the merged BERTopic model
- gets merged topic assignments for every article
- extracts document embeddings
- runs UMAP to obtain `umap_x` and `umap_y`
- creates display-friendly topic labels
- returns:
  - `merged_articles`
  - `merged_topic_info_display`
  - the fitted UMAP reducer

The notebook uses it in Cell 6.

### `plot_merged_topic_umap`

This helper:
- plots all merged articles in 2D
- colors the top merged topics
- groups all non-top topics into a grey background
- writes topic labels near topic centroids

The notebook uses it in Cell 7.

### `plot_outlet_highlight_umap`

This helper:
- draws all articles in light grey
- draws one outlet's articles in red
- computes how many merged topics that outlet covers
- labels the outlet's most represented merged topics
- writes a title like:
  - `Outlet Across Merged Topic Space (x/y merged topics covered)`

The notebook uses it in Cell 8.

### `build_outlet_topic_coverage_summary`

This helper:
- counts how many merged topics each alternative-media outlet covers
- counts how many articles each outlet contributes
- counts how many of an outlet's articles are still assigned to topic `-1`
- computes coverage share over all non-outlier merged topics

The notebook uses it in Cell 8 before plotting.

## What The Notebook Produces

If run successfully, the notebook produces:

1. A printout of resolved model paths.
2. A printout confirming each saved BERTopic model was loaded.
3. A merged BERTopic model object in memory.
4. An optional merged-model save if `SAVE_MERGED_MODEL = True`.
5. A prepared-document summary table by outlet.
6. An article-level merged-topic DataFrame called `merged_articles`.
7. A global UMAP figure of the merged topic space.
8. A topic-coverage summary table for alternative-media outlets.
9. One UMAP figure per alternative-media outlet:
   - all documents grey
   - selected outlet red
   - labels for covered merged topics

## Main Assumptions and Caveats

- The notebook assumes all saved outlet models already exist.
- The notebook assumes all outlet corpora can be rebuilt from local raw data.
- The notebook assumes the helper module is importable from `1a_BERTopic`.
- The notebook does not retrain outlet models.
- The notebook does not fit one pooled BERTopic model from scratch.
- Instead, it merges separately trained outlet models and then applies the merged model to rebuilt prepared documents.

## Short Conceptual Summary

In plain language, the notebook does this:

- load seven outlet-specific BERTopic models
- merge them into one shared topic space
- rebuild all article documents in a consistent prepared form
- assign every article to the merged topic space
- reduce article embeddings to 2D with UMAP
- show:
  - the overall merged topic map
  - how strongly each alternative-media outlet spreads across that shared topic space

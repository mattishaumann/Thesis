# Section 5.1 - topic-quality coherence drop-in

Self-contained input for an LLM that has read access to the `tex/` tree. This file adds a quantitative coherence assessment to the topic-modelling results section, closing the audit hole identified by the reviewer: *"how do you know your topics are good?"*. All numbers are reproducible from `02_TopicModeling/topic_coherence_scores.py`; the persisted result is `data/processed/topic_coherence_scores.csv`.

Style notes for the editor:
- Use LaTeX `--` for en-dashes, `---` for em-dashes (matches the surrounding thesis).
- Sentence case for headers.
- The prose below uses spaced hyphens; convert to `--` if the chapter style demands it.

---

## Slot 1: results paragraph + table at the start of Section 5.1

**Where**: First substantive paragraph of Section 5.1 (the topic-modelling results section, immediately before or after the section's opening "we fitted seven per-outlet models" framing). Search the .tex for `\section{...}` headings near "Topic Modelling Results" or "Section~\ref{sec:results:topics}".

**Action**: Insert the paragraph + table block below as the opening quantitative claim of the section, before the qualitative discussion of representative topics.

**Content (drop-in)**:

```latex
Topic-quality assessment combined qualitative inspection of representative
documents (Section~\ref{sec:topic_inspection}) with a quantitative coherence
score. Table~\ref{tab:cv_coherence} reports $C_V$ coherence
\parencite{roder2015coherence} on the top-10 c-TF-IDF terms per topic,
computed with the \texttt{gensim} \texttt{CoherenceModel} against each model's
own modelling-input corpus; tokenisation matched the BERTopic
\texttt{CountVectorizer} pattern (\texttt{\textbackslash b\textbackslash w\textbackslash w+\textbackslash b}) and used the
project's curated German stop-word list. $C_V$ values above $0.5$ are
conventionally interpreted as topically coherent solutions
\parencite{roder2015coherence}. The seven per-outlet models range from
$0.602$ (Anti-Spiegel, the smallest corpus at $n=565$) to $0.755$ (RT~DE),
and the merged 72-topic cross-outlet model scores $0.727$, indicating that
\texttt{BERTopic.merge\_models()} does not degrade per-topic word coherence
relative to the per-outlet models from which it is built.

\begin{table}[t]
\centering
\caption{$C_V$ topic coherence (top-10 c-TF-IDF terms) for the seven
  per-outlet BERTopic models and the merged 72-topic cross-outlet model.
  Each model is scored against its own modelling-input corpus
  \parencite{roder2015coherence}. The conventional threshold for
  ``coherent'' solutions is $C_V > 0.5$.}
\label{tab:cv_coherence}
\small
\begin{tabular}{l r r r}
\toprule
Model & Documents & Topics & $C_V$ \\
\midrule
Tagesschau          & 6{,}319 & 55 & 0.735 \\
RT DE               & 4{,}559 & 50 & 0.755 \\
Anti-Spiegel        &    565 & 17 & 0.602 \\
Compact             & 1{,}486 & 31 & 0.727 \\
Deutschland-Kurier  & 1{,}465 & 26 & 0.702 \\
NIUS                & 3{,}266 & 38 & 0.754 \\
Tichys Einblick     & 2{,}756 & 51 & 0.675 \\
\midrule
Merged (cross-outlet) & 20{,}416 & 72 & 0.727 \\
\bottomrule
\end{tabular}
\end{table}
```

**Verification numbers** (do not paste; for the editor's sanity-check):

| Outlet | n_docs | n_topics | C_V (raw) |
|---|---:|---:|---:|
| Tagesschau | 6,319 | 55 | 0.7350606095419076 |
| RT DE | 4,559 | 50 | 0.7549963928781754 |
| Anti-Spiegel | 565 | 17 | 0.6015524302684618 |
| Compact | 1,486 | 31 | 0.7271094409233154 |
| Deutschland-Kurier | 1,465 | 26 | 0.7018246266265707 |
| NIUS | 3,266 | 38 | 0.7538429631843172 |
| Tichys Einblick | 2,756 | 51 | 0.6750464641210242 |
| Merged (cross-outlet) | 20,416 | 72 | 0.7268743062161479 |

Source: `data/processed/topic_coherence_scores.csv`. Re-running `python 02_TopicModeling/topic_coherence_scores.py` against the frozen models reproduces these numbers exactly.

---

## Slot 2: BibTeX entry

**Where**: `tex/References.bib` (or the bibliography file the chapter `\bibliography{...}`'s).

**Action**: Append the entry below if not already present.

**Content (drop-in)**:

```bibtex
@inproceedings{roder2015coherence,
  title     = {Exploring the Space of Topic Coherence Measures},
  author    = {R{\"o}der, Michael and Both, Andreas and Hinneburg, Alexander},
  booktitle = {Proceedings of the Eighth ACM International Conference on Web Search and Data Mining (WSDM '15)},
  year      = {2015},
  pages     = {399--408},
  doi       = {10.1145/2684822.2685324}
}
```

---

## Slot 3 (optional): one-line methods sentence

**Where**: Section 4.x of the methodology chapter, in the BERTopic subsection where evaluation criteria are introduced. Search the .tex for the existing prose around "topic quality" or "coherence" in the methodology chapter.

**Action**: Add the one sentence below where the topic-quality evaluation method is first declared. If the methodology chapter does not yet mention coherence, place it next to the sentence that introduces qualitative inspection of representative documents.

**Content (drop-in)**:

```latex
Topic quality was assessed both qualitatively, by inspecting representative
documents and top c-TF-IDF terms per topic, and quantitatively via
$C_V$ coherence \parencite{roder2015coherence} on the top-10 terms;
per-outlet and merged-model values are reported in
Table~\ref{tab:cv_coherence}.
```

---

## Slot 4 (optional): limitations / discussion note

**Where**: Section 5.x discussion of topic-modelling limitations or Section 6.x methodological caveats. The smallest-corpus result (Anti-Spiegel, $C_V=0.602$) is worth one sentence so the reader does not interpret it as a quality failure.

**Action**: Add the one sentence below where corpus-size dependence is discussed.

**Content (drop-in)**:

```latex
Anti-Spiegel's lower $C_V$ score ($0.602$ versus $0.602$--$0.755$ for the other
six outlets) reflects the smaller corpus ($n=565$) on which the
sliding-window co-occurrence statistics underlying $C_V$ are computed; the
score remains comfortably above the conventional $0.5$ threshold and the
representative-document inspection confirmed substantive thematic coherence
of all 17 Anti-Spiegel topics.
```

---

## Reproducibility

- Script: `02_TopicModeling/topic_coherence_scores.py` (mirrors `topic_diversity_scores.py`).
- Output: `data/processed/topic_coherence_scores.csv` (mirrors `topic_diversity_scores.csv`).
- Inputs read: `data/processed/df_combined_with_merged_topics.csv` and the seven `02_TopicModeling/outputs/{outlet}_model/topics.json` plus `merged_all_outlets_model/topics.json`.
- Dependency: `gensim==4.3.2` (verified in `.venv312`; not currently pinned in `requirements.txt`, consider adding).
- Settings: `CoherenceModel(coherence='c_v', topn=10, processes=1)`; bigrams in c-TF-IDF representations (e.g. "social media") are split into unigrams before scoring so they appear in the unigram-tokenised reference corpus.
- Tokenisation: regex `\b\w\w+\b` on lowercased `Title + " " + Text`, with the project's 145-entry German stop-word list (`02_TopicModeling/01_TopicCreation/stopwords_de.py`).

End of file.

# Methodology section 4 - pass 2 drop-in snippets

Self-contained input for an LLM that has read access to `tex/Chapters/04_Methodology.tex` and `tex/References.bib` (or wherever the bibliography lives). Each numbered slot below maps to one TODO comment or one identified issue in the methodology chapter. For each slot, this file gives:

1. **Where** - search cue or line-range hint to locate the slot in the .tex.
2. **Action** - replace / insert / append.
3. **Content** - the LaTeX-ready text to write, verbatim.
4. **Verification numbers** - the underlying numerical source so the editor can sanity-check the prose.

All counts are verified against the frozen pipeline run `02_TopicModeling/outputs/frozen_merged_runs/2026-04-03_merged_v1` and the canonical `data/processed/df_combined_with_merged_topics.csv` / `03_new_topic_assignments.csv` outputs. Outlet-code abbreviations: TS = Tagesschau, RT = RT DE, AS = Anti-Spiegel, CO = Compact, DK = Deutschland-Kurier, NI = NIUS, TE = Tichys Einblick.

Style notes for the editor:
- Use LaTeX `--` for en-dashes and `---` for em-dashes; the surrounding thesis already does this.
- Sentence case for section headers, no title case.
- Do not introduce em-dashes in prose I authored below; I have used spaced hyphens (` - `) which the editor can keep or convert to `--`.

---

## Slot 1: Preprocessing pipeline figure and prose (Comment 1)

**Where**: Section 4.3 (Preprocessing). Search the .tex for `Comment 1` or for the figure placeholder in the preprocessing subsection. The current prose already mentions per-outlet cleaning and a date window; this slot adds (a) a preprocessing figure, (b) a verified prose paragraph with row counts.

**Action**: Insert the figure (TikZ skeleton in Slot 6) and replace the existing row-count prose with the verified version below.

**Content - prose paragraph (drop-in)**:

```latex
The preprocessing pipeline (Figure~\ref{fig:preprocessing_waterfall}) proceeds in
three stages. \textbf{Stage 1} loads each outlet's raw scrape and applies
outlet-specific cleaning (Section~\ref{sec:cleaning_decisions},
Table~\ref{tab:cleaning_decisions}): JSON content extraction for Tagesschau,
``Mehr zum Thema'' tail removal for RT~DE and Compact, podcast and
audio-platform stripping for Anti-Spiegel and Tichys~Einblick, and per-outlet
deduplication on \texttt{Title} and \texttt{Text}. \textbf{Stage 2} restricts
all outlets to the study window 2025-08-01 to 2026-01-31 (inclusive) and
concatenates the seven cleaned frames into a single canonical corpus
(\texttt{df\_combined.csv}, $N=20{,}440$). \textbf{Stage 3} applies the
modelling-time filter (\texttt{min\_text\_chars}=50, \texttt{min\_tokens}=8,
exact-clean-text deduplication) inside \texttt{prepare\_documents\_with\_audit};
24 articles are excluded (12 short~+~few-tokens; 6 too-few-tokens; 6 duplicate
clean documents, all from Deutschland-Kurier), leaving $N=20{,}416$ articles
that enter the seven per-outlet BERTopic models. After merging, 19,812
articles receive a substantive topic assignment (72 topics) and 628 are
retained in the outlier bucket; 72 of these merged topics are aggregated
into 18 thematic clusters by manual coding
(Section~\ref{sec:manual_clustering}).
```

**Content - figure block (drop-in)**:

```latex
\begin{figure}[t]
  \centering
  \input{figures/preprocessing_waterfall.tex} % TikZ skeleton in Slot 6 below
  \caption{Preprocessing pipeline from per-outlet raw scrapes (left) to the
    final modelling corpus and BERTopic analysis frame (right). Edge labels
    report row counts after each step. Outlet-specific cleaning artefacts
    (italicised drop counts) reflect the regex and metadata filters
    documented in Table~\ref{tab:cleaning_decisions}. The cross-outlet
    concatenation uses \texttt{row\_id} as the canonical document key for
    all downstream analysis.}
  \label{fig:preprocessing_waterfall}
\end{figure}
```

**Verification numbers (do not paste into thesis)**:

Per-outlet waterfall (raw $\rightarrow$ post-clean $\rightarrow$ post-window $\rightarrow$ canonical):

| Outlet | Raw | Date filter | Per-outlet dedup / drop | Canonical (`df_combined.csv`) | In modelling input |
|---|---:|---:|---:|---:|---:|
| RT DE | 12,390 | 4,570 | 4,560 | 4,560 | 4,559 |
| Compact | 2,045 | 1,614 | -- | 1,486 | 1,486 |
| NIUS | 4,885 | 3,271 | 3,269 (drop "Show" cat) | 3,269 | 3,266 |
| Tichys | 3,126 | 2,946 | 2,756 (drop magazine + TE-Wecker) | 2,756 | 2,756 |
| Anti-Spiegel | 914 | 613 | 565 (drop podcast/Spotify-VK) | 565 | 565 |
| Tagesschau | 6,320 | 6,320 | 6,320 | 6,320 | 6,319 |
| Deutschland-Kurier | 1,977 | 1,484 | 1,484 | 1,484 | 1,465 |
| **Total** | -- | -- | -- | **20,440** | **20,416** |

Modelling-filter exclusions (24 rows): 12 short + few-tokens; 6 too-few-tokens; 6 duplicate-clean-document (all 6 from DK).

Final analysis frame (`03_new_topic_assignments.csv`): 19,812 non-outlier articles, 628 outliers, 72 topics, 18 manual clusters.

---

## Slot 2: Per-outlet cleaning decisions table (Comment 4)

**Where**: Section 4.3, lines approximately L174-L184 of `04_Methodology.tex`. The existing prose duplicates outlet-by-outlet cleaning rules in narrative form. Replace that block with the table below and a one-sentence pointer.

**Action**: Replace L174-L184 (the duplicated narrative) with one-sentence intro + the `tabularx` table.

**Content (drop-in)**:

```latex
Outlet-specific cleaning decisions arose from inspection of an exploratory
BERTopic run on uncleaned text and target boilerplate, structural artefacts,
and non-editorial formats that would otherwise have formed spurious topics
(Table~\ref{tab:cleaning_decisions}).

\begin{table}[t]
\centering
\caption{Outlet-specific preprocessing decisions and the artefacts they
  remove. Row counts report items dropped at the canonical loader stage;
  ``all'' indicates a per-document text transformation rather than row
  removal.}
\label{tab:cleaning_decisions}
\small
\begin{tabularx}{\textwidth}{@{} l X X r X @{}}
\toprule
\textbf{Outlet} & \textbf{Problem detected} & \textbf{Preprocessing action} & \textbf{Rows} & \textbf{Rationale} \\
\midrule
Tagesschau & \texttt{content} stored as JSON array of typed blocks (text, headline, image, list) with HTML markup inside text values & Parse JSON; keep only \texttt{type=text} and \texttt{type=headline}; HTML-unescape and strip tags; collapse whitespace & all 6{,}320 & Without parsing, BERTopic embeds JSON keys, not prose \\
\addlinespace
RT DE & Recurring ``Mehr zum Thema'' link block appended to article body; large legacy archive outside study window & Regex strip from \texttt{\textbackslash n*\textbackslash s*Mehr zum Thema} to end; date-window filter; dedup on \texttt{Title} OR \texttt{Text} & 7{,}830 & Tail block formed a related-article topic in the pilot run \\
\addlinespace
Anti-Spiegel & ``Tacheles \#'' podcast announcements; podcast and Spotify-VK promotion paragraphs; ``DD.\ Monat YYYY HH:MM Uhr'' timestamp prefixes & Drop rows where \texttt{Title} contains ``Tacheles \#''; drop rows whose body contains \texttt{Anti-Spiegel-Podcast}, ``Den Podcast können Sie hier'', or (\texttt{spotify} $\land$ \texttt{VK}); leading-timestamp regex via ``Teile diesen Beitrag'' split & 48 dropped & Podcast pages would form a single dominant audience-acquisition cluster with no editorial signal \\
\addlinespace
Compact & Trailing ``Mehr zum Thema'' block; recurring shop / merch / ``Hier bestellen'' / YouTube-bell promotional copy embedded mid-article & Regex strip ``Mehr zum Thema'' tail; \texttt{max\_df=0.85}; outlet-specific stop words (\texttt{compact, tv, youtube, kanal, glocke, bestellen, magazin, heft, shop}) & all & Without stop-word suppression, c-TF-IDF labels for unrelated topics were dominated by merchandise vocabulary \\
\addlinespace
Deutschland-Kurier & Missing \texttt{Text} for some scraped rows; near-duplicate (\texttt{Title}, \texttt{Text}) pairs from re-publishing & Drop NaN-text rows; \texttt{drop\_duplicates(subset=[Title, Text])}; date-window filter & 154 NaN; 1 dup; 338 outside window & Empty bodies inflate token counts but carry no signal; near-dups create artificial topic mass \\
\addlinespace
NIUS & \texttt{authors} stored as JSON-like \texttt{\{'name': ...\}}; ``Show'' category = celebrity / sponsored video format; trailing ``Mehr / Auch / Lesen Sie auch bei NIUS:'' navigation block & Regex-extract author name(s); drop rows where \texttt{Categories} contains ``Show''; regex strip trailing reference block & 2 (Show) + tail strip on all rows & Show category formed a celebrity-promo cluster in the pilot run \\
\addlinespace
Tichys Einblick & Magazine-issue sale stubs (``Tichys Einblick MM-YYYY: \ldots'', URL slug \texttt{/daili-es-sentials/tichys-einblick-MM-YYYY}, ``direkt als PDF erhältlich''); TE-Wecker daily-roundup posts; SoundCloud, Spotify, iTunes embed boilerplate in body & Regex drop magazine-issue rows; regex drop TE-Wecker rows; outlet stop words (\texttt{soundcloud, spotify, itunes, amazon, abonnieren, \ldots}); \texttt{reduce\_outliers} threshold lowered to 0.05 & 190 dropped & Magazine and roundup formats clustered as a publication-format topic in the pilot run \\
\bottomrule
\end{tabularx}
\end{table}
```

**Verification**: each row count matches `02_TopicModeling/merged_outlets_analysis.py` loaders re-executed on raw data; "all" entries are per-document transformations.

---

## Slot 3: Per-outlet vs pooled rationale (Comment 2a, `%% Add multimodel`)

**Where**: Section 4.x where BERTopic is first introduced. Search the .tex for the comment `%% Add multimodel` (Comment 2a in your audit). The slot needs 3-5 sentences explaining why per-outlet then merge rather than one pooled model.

**Action**: Replace the placeholder comment with the paragraph below.

**Content (drop-in)**:

```latex
A single BERTopic model fitted to the full 20{,}440-article corpus was rejected
on principled grounds rather than tested as an empirical baseline. Tagesschau
and RT~DE together account for $53\%$ of the corpus, and a pooled embedding
step would let their editorial vocabulary anchor the cluster centroids;
distinctive themes from the smaller outlets (Anti-Spiegel at $2.8\%$ of the
corpus) would either be absorbed into mainstream clusters or assigned to the
HDBSCAN outlier class. The per-outlet-then-merge design recovers each outlet's
topic structure on its own terms before reconciling it in a shared
vocabulary, and it allows HDBSCAN parameters (notably
\texttt{min\_cluster\_size}) to be scaled to each outlet's corpus size rather
than averaged across a four-orders-of-magnitude size gradient. Merging via
\texttt{BERTopic.merge\_models()} \parencite{grootendorst2022} operates on
c-TF-IDF topic embeddings, so reconciliation reduces to a single tunable
parameter (\texttt{min\_similarity}=0.70) rather than a sequence of editorial
judgements. The cumulative merge collapses 268 per-outlet source topics to 72
merged topics, with Deutschland-Kurier (added last) contributing zero net new
topics, indicating that the shared topic space was already saturated.
```

**Verification**:
- Corpus shares: TS 6,320/20,440 = 30.9%; RT 4,560/20,440 = 22.3%; combined 53.2%; AS 565/20,440 = 2.8%.
- 268 source topics: TS 55 + RT 50 + AS 17 + TE 51 + NI 38 + CO 31 + DK 26 = 268.
- Merge path documented in `02_TopicModeling/BERTOPIC_THESIS_TABLES.md` lines 358-364.
- DK net contribution: +0 topics (TS through CO already at 72).

---

## Slot 4: Topic-diversity / class-imbalance methodological framing (Comment 2b, `%ADD sth. wrt. Topic Diversity`)

**Where**: Section 4.x on the topic-diversity / JSD methodology. Search the .tex for the comment `%ADD sth. wrt. Topic Diversity` (Comment 2b). The slot needs a one-paragraph methodological framing for why normalised Shannon entropy is reported alongside JSD.

**Action**: Replace the placeholder with the paragraph below. The optional decomposition table can either go inline or in an appendix; I suggest the appendix.

**Content (drop-in)**:

```latex
Because Anti-Spiegel concentrates $50.5\%$ of its articles in three
Russia-Ukraine topics and covers only 43 of the 72 topics, its Jensen-Shannon
divergence from Tagesschau ($\mathrm{JSD}=0.279$) could in principle be
inflated by either the over-coverage of those three topics or the absence of
mainstream topics from its agenda. We therefore report normalised Shannon
entropy $H/\ln K$ alongside JSD: entropy captures within-outlet concentration
as a quantity that is not confounded with topic-set asymmetry, while JSD
captures directional divergence from the mainstream reference. A topic-level
decomposition (Appendix~\ref{app:jsd_decomp}) shows that $32\%$ of
$\mathrm{JSD}(\mathrm{AS}, \mathrm{TS})$ comes from the three over-covered
topics and $41\%$ from the 29 mainstream topics absent from Anti-Spiegel's
coverage; capping Anti-Spiegel's top-1 share at $5\%$ reduces JSD only
marginally (from $0.279$ to $0.275$), confirming that the divergence reflects
genuine agenda asymmetry rather than concentration alone.
```

**Optional appendix block** (drop into Appendix `\section{JSD decomposition}`):

```latex
\section{Decomposition of $\mathrm{JSD}(\mathrm{Anti\text{-}Spiegel}, \mathrm{Tagesschau})$}
\label{app:jsd_decomp}

JSD is additive across topics:
$\mathrm{JSD}(P,Q) = \tfrac{1}{2}\,\mathrm{KL}(P\,\Vert\,M) + \tfrac{1}{2}\,\mathrm{KL}(Q\,\Vert\,M)$
with $M = \tfrac{1}{2}(P+Q)$, and each KL term decomposes as a sum of per-topic
contributions $p_k \log(p_k / m_k)$. Table~\ref{tab:jsd_decomp} reports the per-topic
contributions for the six topics with the largest contribution to the total
$\mathrm{JSD}(\mathrm{AS},\mathrm{TS}) = 0.2794$.

\begin{table}[h]
\centering
\caption{Per-topic contributions to $\mathrm{JSD}(\mathrm{Anti\text{-}Spiegel}, \mathrm{Tagesschau})$.}
\label{tab:jsd_decomp}
\small
\begin{tabular}{l r r r}
\toprule
Topic & AS \% & TS \% & Contribution \\
\midrule
Topic 3 -- vermögenswerte\_nato\_belgien\_russlands       & 17.30 & 1.62 & 0.0379 \\
Topic 1 -- selenskyj\_moskau\_nato\_wladimir              & 20.22 & 4.64 & 0.0263 \\
Topic 7 -- ukrainischen\_ukrainische\_kiew\_streitkräfte  & 12.93 & 1.54 & 0.0256 \\
Topic 53 -- selensky\_nabu\_leser\_bellingcat             &  7.29 & 0.11 & 0.0227 \\
Topic 31 -- dax\_anleger\_aktien\_punkten                 &  0.00 & 3.05 & 0.0106 \\
Topic 21 -- patienten\_studien\_medikamente\_ärzte        &  0.00 & 2.91 & 0.0101 \\
\midrule
\multicolumn{3}{l}{AS top-3 over-covered topics (sum)}    & 0.0899 \\
\multicolumn{3}{l}{Topics with $p_{\mathrm{AS}}=0$, $p_{\mathrm{TS}}>0$ (29 topics)} & 0.1155 \\
\multicolumn{3}{l}{Remaining cross-coverage}              & 0.0740 \\
\midrule
\multicolumn{3}{l}{\textbf{Total $\mathrm{JSD}(\mathrm{AS},\mathrm{TS})$}}            & \textbf{0.2794} \\
\bottomrule
\end{tabular}
\end{table}

A sensitivity check capping Anti-Spiegel's top-1 share at $10\%$ and $5\%$ and
renormalising yields $\mathrm{JSD} = 0.272$ and $0.275$ respectively, confirming
that JSD is robust to concentration in the top of the distribution and is
driven primarily by the topics absent from Anti-Spiegel's coverage.
```

**Verification numbers**:

| Outlet | Top-1 % | Top-3 % | Top-5 % | Topics covered | Topics with mass <1% | Topics with mass <0.5% |
|---|---:|---:|---:|---:|---:|---:|
| Tagesschau | 6.41 | 14.10 | 19.96 | 72 | 24 | 15 |
| Tichys Einblick | 6.45 | 18.32 | 25.27 | 72 | 36 | 19 |
| NIUS | 6.13 | 17.98 | 27.47 | 70 | 41 | 21 |
| Compact | 7.57 | 19.90 | 30.64 | 69 | 39 | 24 |
| Deutschland-Kurier | 9.82 | 23.19 | 34.42 | 67 | 41 | 26 |
| RT DE | 12.34 | 32.02 | 41.77 | 72 | 45 | 28 |
| Anti-Spiegel | 20.22 | 50.46 | 62.66 | 43 | 28 | 15 |

JSD pairwise vs Tagesschau (log base 2, divergence in [0, 1]): RT 0.108; TE 0.102; NI 0.142; CO 0.178; DK 0.162; AS 0.279.

JSD sensitivity: AS top-1 capped at 10% $\to$ 0.272; capped at 5% $\to$ 0.275; AS top-3 zeroed and renormalised $\to$ 0.248.

---

## Slot 5: References.bib additions and (XXX) citation insertions

**Where**: `tex/References.bib` (or wherever the `.bib` is). Two `(XXX)` citation slots in `04_Methodology.tex`: one near the JSD definition, one near the topic-diversity / Bianchi formula.

**Action**: Append the BibTeX entries below to the `.bib`. Replace the two `(XXX)` markers with the indicated `\parencite{...}` calls.

**Content - BibTeX entries**:

```bibtex
@article{grootendorst2022,
  title   = {{BERTopic}: Neural topic modeling with a class-based {TF-IDF} procedure},
  author  = {Grootendorst, Maarten},
  year    = {2022},
  journal = {arXiv preprint arXiv:2203.05794},
  doi     = {10.48550/arXiv.2203.05794},
  url     = {https://arxiv.org/abs/2203.05794}
}

@inproceedings{reimers2019sentencebert,
  title     = {{Sentence-BERT}: Sentence Embeddings using {Siamese BERT}-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year      = {2019},
  doi       = {10.18653/v1/D19-1410}
}

@inproceedings{reimers2020multilingual,
  title     = {Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2020},
  doi       = {10.18653/v1/2020.emnlp-main.365}
}

@inproceedings{campello2013hdbscan,
  title     = {Density-Based Clustering Based on Hierarchical Density Estimates},
  author    = {Campello, Ricardo J. G. B. and Moulavi, Davoud and Sander, J{\"o}rg},
  booktitle = {Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year      = {2013},
  doi       = {10.1007/978-3-642-37456-2_14}
}

@article{mcinnes2017hdbscan,
  title   = {hdbscan: Hierarchical density based clustering},
  author  = {McInnes, Leland and Healy, John and Astels, Steve},
  journal = {Journal of Open Source Software},
  volume  = {2},
  number  = {11},
  pages   = {205},
  year    = {2017},
  doi     = {10.21105/joss.00205}
}

@article{mcinnes2018umap,
  title   = {{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author  = {McInnes, Leland and Healy, John and Melville, James},
  year    = {2018},
  journal = {arXiv preprint arXiv:1802.03426},
  doi     = {10.48550/arXiv.1802.03426}
}

@article{blei2003lda,
  title   = {Latent {Dirichlet} Allocation},
  author  = {Blei, David M. and Ng, Andrew Y. and Jordan, Michael I.},
  journal = {Journal of Machine Learning Research},
  volume  = {3},
  pages   = {993--1022},
  year    = {2003}
}

@inproceedings{bianchi2021crosslingual,
  title     = {Cross-lingual Contextualized Topic Models with Zero-shot Learning},
  author    = {Bianchi, Federico and Terragni, Silvia and Hovy, Dirk and Nozza, Debora and Fersini, Elisabetta},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2021},
  doi       = {10.18653/v1/2021.eacl-main.143}
}

@article{lin1991jsd,
  title   = {Divergence Measures Based on the {Shannon} Entropy},
  author  = {Lin, Jianhua},
  journal = {IEEE Transactions on Information Theory},
  volume  = {37},
  number  = {1},
  pages   = {145--151},
  year    = {1991},
  doi     = {10.1109/18.61115}
}

@article{shannon1948,
  title   = {A Mathematical Theory of Communication},
  author  = {Shannon, Claude E.},
  journal = {The Bell System Technical Journal},
  volume  = {27},
  pages   = {379--423, 623--656},
  year    = {1948},
  doi     = {10.1002/j.1538-7305.1948.tb01338.x}
}
```

**Content - (XXX) replacements in `04_Methodology.tex`**:

- The `(XXX)` near the JSD formal definition should become `\parencite{lin1991jsd}`. Optionally pair with `\parencite{shannon1948}` if the surrounding sentence introduces Shannon entropy first.
- The `(XXX)` near the topic-diversity formula `|unique words| / (10 * n_active_topics)` should become `\parencite{bianchi2021crosslingual}`. The formula is referenced verbatim in `02_TopicModeling/topic_diversity_scores.py:4`.

**Caveat for the editor**: the `Topic_Diversity_Analysis.ipynb` notebook references "Bonanomi et al., 2019, *Physica A*" for raw Shannon entropy in nats as a media-diversity measure. I could not locate this paper (no matching paper exists with that exact metadata). Do not cite it without first verifying. If a media-diversity citation is needed alongside Shannon's original, prefer Boczkowski \& Mitchelstein (2013) *The News Gap* or a comparable journalism-studies reference; flag this back to the user rather than papering over it.

---

## Slot 6: TikZ skeleton for the preprocessing figure

**Where**: Create a new file `tex/figures/preprocessing_waterfall.tex` (or wherever `\input{figures/...}` resolves). The figure environment in Slot 1 references it.

**Action**: Create the file with the skeleton below; flesh out cosmetics as needed.

**Content (drop-in)**:

```latex
% preprocessing_waterfall.tex
% Two-column waterfall: per-outlet preparation (left), modelling-time corpus (right).
% Tested with TikZ 3.x; relies on the positioning, arrows.meta, and shapes.geometric libraries.

\begin{tikzpicture}[
  font=\footnotesize,
  >={Stealth[length=2mm]},
  outlet/.style={draw, rounded corners=2pt, minimum width=58mm, minimum height=7mm,
                 align=left, fill=blue!4},
  corpus/.style={draw, rounded corners=3pt, minimum width=42mm, minimum height=10mm,
                 align=center, fill=gray!8, font=\small},
  drop/.style ={font=\scriptsize\itshape, gray!70!black},
  countlbl/.style={font=\scriptsize, midway, above, sloped},
  node distance=4mm and 12mm,
]

% --- Left column: per-outlet swim lanes ---
\node[outlet] (rt)   {\textbf{RT DE} \hfill 12{,}390 $\to$ 4{,}570 $\to$ \textbf{4{,}560}};
\node[outlet, below=of rt]   (co) {\textbf{Compact} \hfill 2{,}045 $\to$ 1{,}614 $\to$ \textbf{1{,}486}};
\node[outlet, below=of co]   (ni) {\textbf{NIUS} \hfill 4{,}885 $\to$ 3{,}271 $\to$ \textbf{3{,}269}};
\node[outlet, below=of ni]   (te) {\textbf{Tichys Einblick} \hfill 3{,}126 $\to$ 2{,}946 $\to$ \textbf{2{,}756}};
\node[outlet, below=of te]   (as) {\textbf{Anti-Spiegel} \hfill 914 $\to$ 613 $\to$ \textbf{565}};
\node[outlet, below=of as]   (ts) {\textbf{Tagesschau} \hfill 6{,}320 $\to$ 6{,}320 $\to$ \textbf{6{,}320}};
\node[outlet, below=of ts]   (dk) {\textbf{Deutschland-Kurier} \hfill 1{,}977 $\to$ 1{,}484 $\to$ \textbf{1{,}484}};

% --- Right column: corpus stages ---
\node[corpus, right=24mm of ni, anchor=west] (combined)
  {\texttt{df\_combined.csv}\\$N=20{,}440$};
\node[corpus, below=8mm of combined]         (filter)
  {Modelling filter\\$\geq 50$ chars, $\geq 8$ tokens,\\dedup clean text\\($-24$ rows)};
\node[corpus, below=8mm of filter]           (bertopic)
  {7 per-outlet\\BERTopic models\\(268 source topics)};
\node[corpus, below=8mm of bertopic]         (merge)
  {\texttt{merge\_models}\\$\min\_\mathrm{similarity}=0.70$\\TS anchor $\to$ 72 topics};
\node[corpus, below=8mm of merge]            (final)
  {Final analysis frame\\19{,}812 + 628 outlier\\72 topics, 18 clusters};

% --- Fan-in arrows from outlets to combined corpus ---
\foreach \src in {rt, co, ni, te, as, ts, dk}
  \draw[->] (\src.east) -- (combined.west);

% --- Vertical chain ---
\draw[->] (combined) -- (filter)   node[countlbl]{$\to 20{,}416$};
\draw[->] (filter)   -- (bertopic);
\draw[->] (bertopic) -- (merge)    node[countlbl]{$268 \to 72$};
\draw[->] (merge)    -- (final);

% --- Optional study-window watermark ---
\node[font=\scriptsize\itshape, gray, above=2mm of rt, anchor=south]
  {Study window: 2025-08-01 / 2026-01-31};

\end{tikzpicture}
```

**Notes for the editor**:
- Outlet-specific cleaning callouts can be added as small `\node[drop, above of=...]` italic labels per swim lane, e.g. above RT: *"+ strip Mehr zum Thema"*; above Tagesschau: *"+ JSON content extraction"*.
- The 7-into-1 fan-in produces dense arrows. If too cluttered, replace with a single junction node (e.g. `\node[circle, fill=black, inner sep=1pt] (j) at ($(...)+(0,0)$) {};`) and route each outlet through it.
- All bold counts on the left match the canonical `df_combined.csv` totals.

---

## Appendix A: full per-outlet waterfall (verification only, do not paste)

| Outlet | Step | In | Out |
|---|---|---:|---:|
| RT DE | Raw load (`RT_de.xlsx`) | -- | 12,390 |
| | Drop `Full_Text`; whitespace collapse | 12,390 | 12,390 |
| | Strip "Mehr zum Thema" tail | 12,390 | 12,390 |
| | Date filter [2025-08-01, 2026-02-01) | 12,390 | 4,570 |
| | Dedup on Title OR Text | 4,570 | **4,560** |
| Compact | Raw load (214 daily CSVs) | -- | 2,045 |
| | Date filter + col select + Mehr-zum-Thema | 2,045 | 1,614 |
| | Concat-time consolidation (canonical) | 1,614 | **1,486** |
| NIUS | Raw load (285 daily CSVs) | -- | 4,885 |
| | Col rename + drop NaN | 4,885 | 4,872 |
| | Dedup on Title OR Text | 4,872 | 4,868 |
| | Date filter; strip Mehr/Auch bei NIUS | 4,868 | 3,271 |
| | Drop Categories=="Show" | 3,271 | **3,269** |
| Tichys Einblick | Raw load (195 daily CSVs) | -- | 3,126 |
| | Date filter | 3,126 | 2,946 |
| | Drop magazine-issue + TE-Wecker rows | 2,946 | **2,756** |
| Anti-Spiegel | Raw load (307 daily CSVs) | -- | 914 |
| | Date filter | 914 | 613 |
| | Drop "Tacheles #" titles | 613 | 589 |
| | Drop podcast / Spotify-VK promo | 589 | **565** |
| Tagesschau | Raw load (2 quarterly CSVs) | -- | 6,320 |
| | JSON-extract content (text + headline) | 6,320 | 6,320 |
| | HTML unescape + tag strip | 6,320 | 6,320 |
| | Date filter (Berlin tz) | 6,320 | **6,320** |
| Deutschland-Kurier | Raw load (222 daily CSVs) | -- | 1,977 |
| | Drop NaN Text | 1,977 | 1,823 |
| | Dedup on (Title, Text) | 1,823 | 1,822 |
| | Date filter | 1,822 | **1,484** |

Cross-outlet:

| Step | In | Out |
|---|---:|---:|
| Concat 7 cleaned frames | 20,440 | 20,440 |
| Modelling filter (chars $\geq$ 50, tokens $\geq$ 8, dedup-clean) | 20,440 | **20,416** |
| 7 per-outlet BERTopic fits | 20,416 | 20,416 |
| `merge_models(min_similarity=0.70)` | 268 source topics | 72 merged |
| Final analysis frame | 20,440 | 19,812 non-outlier + 628 outlier |

---

## Appendix B: source files for the editor's reference

- Per-outlet canonical loaders: `02_TopicModeling/merged_outlets_analysis.py:203-393` (one function per outlet).
- Modelling filter: `02_TopicModeling/bertopic_pipeline.py` (`prepare_documents_with_audit`).
- Merged-model build: `02_TopicModeling/Merged_BERTopic_Step_By_Step.ipynb`.
- Manual 72-to-18 cluster mapping: `02_TopicModeling/outputs/03_manual_topiclabels_to_assignments.ipynb` (input `02_topic_groupings.csv`, output `03_new_topic_assignments.csv`).
- Topic-diversity formula reference: `02_TopicModeling/topic_diversity_scores.py:4`.
- JSD computation reference: `02_TopicModeling/Topic_Diversity_Analysis.ipynb` (`scipy.spatial.distance.jensenshannon(P, Q, base=2) ** 2`).
- Frozen merged run snapshot: `02_TopicModeling/outputs/frozen_merged_runs/2026-04-03_merged_v1`.
- Pre-existing methodology source markdown: `02_TopicModeling/BERTOPIC_METHODOLOGY.md` and `02_TopicModeling/BERTOPIC_THESIS_TABLES.md` (LaTeX tables A, B, C already drafted there for shared config, per-outlet overrides, merge construction).

End of file.

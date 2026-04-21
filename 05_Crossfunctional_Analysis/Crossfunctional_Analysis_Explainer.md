# Crossfunctional Analysis Explainer (Easy Version)

This file explains what was done in the new crossfunctional run, pillar by pillar (RQ1, RQ2, RQ3), plus the cross-mechanism synthesis.

All results are based only on these four input files in `outputs/`:
- `03_new_topic_assignments.csv`
- `emotion_fulltext_with_topiclabels.csv`
- `emotion_mainstream_resultsv3.csv`
- `media_framing_final_run_5_4mini_classifications.csv`

## 0) What was built

I created:
- one reproducible analysis pipeline: `build_crossfunctional_deep_analysis.py`
- one master notebook: `01_crossfunctional_deep_analysis.ipynb`
- one integrated dataset for analysis: `outputs/crossfunctional_integrated_dataset.csv`
- thesis-ready visuals in `figures/`
- thesis-ready tables in `tables/`

Technical integration steps:
- harmonized source names (`RT` -> `RT_de`, `Tichys Einblick` -> `Tichys_Einblick`)
- used `source + row_id` as article-level join key
- treated framing data as hit-level and aggregated it to article level (`delegit_hit_share`, hit counts, category shares)
- kept full-text emotion scores as primary for affective analysis

---

## 1) RQ1 - Agenda Divergence

### What was done
- Built topic distributions per outlet from `Manual Topic Label`.
- Compared each outlet to Tagesschau using:
  - topic amplification/omission (`log2_amplification_vs_tagesschau`)
  - Jensen-Shannon divergence (JSD)
  - Spearman rank correlation
  - normalized topic entropy (coverage breadth/diversity)

### Visualizations
- `figures/rq1_topic_amplification_heatmap.png`  
  Shows where outlets amplify or under-cover topics relative to Tagesschau.
- `figures/rq1_jsd_bar.png`  
  Outlet-level divergence from Tagesschau.
- `figures/rq1_jsd_entropy_scatter.png`  
  Relationship between diversity breadth and divergence.

### Main findings
- Strongest agenda divergence from Tagesschau:
  1. `Antispiegel` (JSD = 0.403)
  2. `Compact` (JSD = 0.257)
  3. `Deutschlandkurier` (JSD = 0.233)
- This indicates systematic agenda distance, not just random variation.
- Topic-level amplification/omission patterns are exported in:
  - `tables/rq1_topic_amplification_top_bottom.csv`

---

## 2) RQ2 - Delegitimization

### What I did
- Used framing categories at hit level.
- Defined delegitimization as share of hits in:
  - `VERZERRUNG/MANIPULATION`
  - `POSITIONS-/PARTEILICHKEITS-BIAS`
  - `DISINFORMATION/FALSCHDARSTELLUNG`
  - `VERSAGEN/INKOMPETENZ`
- Computed:
  - outlet-level category composition
  - shifts vs Tagesschau
  - topic-level delegitimization intensity

### Visualizations
- `figures/rq2_category_stacked.png`  
  Category mix by outlet.
- `figures/rq2_slope_vs_tagesschau.png`  
  How each outlet deviates from Tagesschau category shares.
- `figures/rq2_topic_delegit_heatmap.png`  
  Topic-level delegitimization intensity by outlet.

### Main findings
- Delegitimization share (higher = more delegitimizing framing among mainstream references):
  - `Antispiegel`: 0.539
  - `Tichys_Einblick`: 0.489
  - `Nius`: 0.380
  - `Tagesschau`: 0.007
- Interpretation: alternative outlets differ not only in *how often* they reference mainstream actors, but especially in *how negatively* those references are framed.

---

## 3) RQ3 - Affective Mobilization

### What I did
- Used full-text emotion scores as primary.
- Built AMI (Affective Mobilization Index):
  - `AMI = (z(anger) + z(fear)) / 2`
- Added sensitivity variant:
  - `ami_sensitivity_raw_mean = (anger + fear) / 2`
- Calculated outlet-level means with bootstrap confidence intervals.
- Added topic-level and monthly trend analysis.

### Visualizations
- `figures/rq3_ami_violin.png`  
  Distribution comparison of AMI by outlet (rendered as robust box-style plot).
- `figures/rq3_monthly_trends.png`  
  Monthly AMI trends by outlet group.
- `figures/rq3_topic_ami_heatmap.png`  
  Topic-outlet heatmap of mean AMI.

### Main findings
- Highest AMI outlets:
  - `Compact`: AMI = 0.331 (95% CI: 0.280 to 0.382)
  - `Tichys_Einblick`: AMI = 0.330 (95% CI: 0.303 to 0.356)
- Anger is generally dominant, with fear contributing additional variation in mobilization intensity.
- Topic-level affective differences are substantial and not uniform across outlets.

---

## 4) Cross-Mechanism Synthesis (RQ1 + RQ2 + RQ3)

### What I did
- Built a topic-outlet mechanism matrix combining:
  - agenda amplification (`log2_amplification_vs_tagesschau`)
  - delegitimization intensity (`delegit_share`)
  - affective mobilization (`ami_mean`)
- Standardized each component (z-scores) and combined into:
  - `convergent_score = z_amp + z_delegit + z_ami`
- Estimated two interpretable association models:
  - `AMI ~ delegit_hit_share + outlet_group + topic_cluster`
  - `amplification ~ delegit_share + ami_mean + outlet_group` (weighted by topic-article volume)

### Visualizations
- `figures/synthesis_bubble_plot.png`  
  X = amplification, Y = delegitimization, color = AMI, size = article volume.

### Main findings
- Top convergent mechanism cells include:
  - `Antispiegel` × `Corruption Journalism`
  - `Compact` × `Corruption Journalism`
  - `Tichys_Einblick` × `Green Politics & Woke Culture#`
  - `Tichys_Einblick` × `German TV, Talk Shows & ÖRR Criticism`
- These are high-priority thesis cases where agenda divergence, delegitimization, and emotional mobilization co-occur strongly.

Related output files:
- `tables/synthesis_topic_outlet_matrix.csv`
- `tables/synthesis_convergent_mechanism_shortlist.csv`
- `tables/synthesis_model_ami_vs_delegit.txt`
- `tables/synthesis_model_amplification_vs_mechanisms.txt`

---

## 5) How to use this for thesis writing

Recommended narrative flow:
1. Start with RQ1 (agenda distance from Tagesschau).
2. Show RQ2 (tone of mainstream references, not just reference volume).
3. Add RQ3 (affective intensity, especially anger/fear profile).
4. Conclude with synthesis: identify specific topic-outlet zones where all three mechanisms are jointly strong.

If you want, I can now also produce a second markdown file that is ready to paste directly into your thesis results chapter (more academic wording and fewer technical terms).

# Merged BERTopic Merge Verdict

Status: proceed to article-level assignment.

## Overall verdict

The cumulative merge behaved well. The final merged model has **72 substantive topics** (excluding `-1`) across all seven outlets. That is a strong compression from the outlet-specific source models and does not show signs of topic-space explosion.

At this stage I would:

- keep `MIN_SIMILARITY = 0.7`
- keep the current per-outlet models as they are
- not rename topics yet
- not revisit merge settings yet
- move on to article-level assignment first
- only decide on merged outlier reduction after inspecting article-level assignments and outlet-topic distributions

## Source topic counts

| Outlet | Source topics |
|---|---:|
| Tagesschau | 55 |
| RT | 50 |
| Anti-Spiegel | 17 |
| Tichys Einblick | 51 |
| Nius | 38 |
| Compact | 31 |
| Deutschlandkurier | 26 |

Total source topics across outlet models: **268**

Final merged substantive topics: **72**

This means the merge collapsed a large amount of overlap rather than preserving outlet-specific fragmentation.

## Cumulative merge path

| Step | Added outlet | Added outlet topics | Merged topics after step | Net new merged topics | Verdict |
|---|---|---:|---:|---:|---|
| Base | Tagesschau | 55 | 55 | baseline | Reasonable broad base; substantively plausible |
| 1 | RT | 50 | 57 | +2 | Strong overlap with Tagesschau; good shared space |
| 2 | Anti-Spiegel | 17 | 58 | +1 | Anti-Spiegel retune worked; no fragmentation problem |
| 3 | Tichys Einblick | 51 | 64 | +6 | Adds distinct themes, but still controlled |
| 4 | Nius | 38 | 67 | +3 | Mostly integrates into existing structure |
| 5 | Compact | 31 | 72 | +5 | Adds some distinct clusters, still acceptable |
| 6 | Deutschlandkurier | 26 | 72 | +0 | Fully absorbed into the merged topic space |

## Interpretation

- **Tagesschau as the broadest base is defensible.** Given the outlet and corpus size, it is substantively plausible that it anchors the broadest topic spectrum.
- **RT integrated strongly.** It did not create a second competing macro-structure.
- **Anti-Spiegel is no longer a comparability concern.** After the retune it contributed only `+1` net merged topic.
- **Tichys and Compact each add some distinct themes.** That is useful differentiation, not a warning sign by itself.
- **Nius mostly folds into the common space.**
- **Deutschlandkurier adds no net new topics.** That is a strong sign that the merged space is already stable.

## What not to do yet

Do not do these before article-level assignment:

- topic renaming
- merged outlier reduction
- further `min_similarity` tuning

Those decisions should be based on article-level assignments, outlier rates, and outlet-topic dominance patterns, not just on the merged model-side topic list.

## Recommended next step

Proceed to:

1. assign the final merged model back to the prepared article corpus
2. inspect the merged article-level outlier count
3. build the outlet-by-topic count pivot table
4. use that table to see whether some topics are genuinely shared or dominated by one outlet
5. only then decide whether merged outlier reduction or topic renaming is needed

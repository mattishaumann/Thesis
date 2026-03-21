# Research Hypotheses & Status

**Thesis**: German alternative media systematically construct a distorted political reality posing a structural threat to democratic discourse.

| # | Hypothesis | Pillar | Methods | Status | Notebook |
|---|---|---|---|---|---|
| H1 | Alt media focus on a narrower topic set to steer conversation | Agenda Distortion | BERTopic, JSD, UMAP, LLM classification | In progress | `experiments/agenda_distortion/` |
| H2 | Alt media portray democratic institutions as corrupt/failing | Delegitimization | NER, Framing Analysis | Pending compute | — |
| H3 | Alt media use anger-dominant rhetoric vs mainstream | Affective Mobilization | Sentiment Analysis, LLM emotion classification | Pending compute | — |

**Workflow**: For each hypothesis: ground in literature → run → evaluate KPIs → validate robustness → document in `findings.md` → only proceed to next hypothesis when current one is resolved.

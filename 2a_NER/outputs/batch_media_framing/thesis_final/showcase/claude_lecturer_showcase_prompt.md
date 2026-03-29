Use the uploaded CSV files:

- `media_framing_lecturer_claude_input.csv`
- `media_framing_lecturer_overview.csv`
- `media_framing_lecturer_source_mix.csv`
- `media_framing_lecturer_output_template.csv`

Task:

Create a short, lecturer-facing showcase CSV that helps a reader who has never really seen the data understand:

1. what we classified,
2. what the category labels mean,
3. what the evidence span is doing,
4. why each example received its label,
5. what the overall result pattern looks like.

Important context from the underlying run:

- The full results file contains `11,973` classified hits.
- These hits come from `7` source outlets and `7,204` unique article rows.
- Overall label counts are:
  - `NEUTRAL`: `8,568`
  - `POSITIONS-/PARTEILICHKEITS-BIAS`: `1,409`
  - `VERZERRUNG/MANIPULATION`: `1,084`
  - `DISINFORMATION/FALSCHDARSTELLUNG`: `377`
  - `VERSAGEN/INKOMPETENZ`: `322`
  - `IRRELEVANT`: `213`
- The curated input CSV is already a short showcase sample: `2` examples per category, `12` rows total.

Your job is not to re-sample the full dataset. Your job is to enrich the `12` uploaded showcase rows so they become understandable for a lecturer.

Output requirements:

- Return only CSV content, no prose before or after it.
- Use exactly the rows from `media_framing_lecturer_claude_input.csv`.
- Keep the same `example_id` values and row count.
- Use the exact column order from `media_framing_lecturer_output_template.csv`.
- Do not add or remove columns.
- Do not invent facts that are not supported by the German title, evidence span, excerpt, category definition, or overview files.
- Keep the writing compact and clear. This is for presentation, not for a methods appendix.

Target output columns and how to fill them:

- `example_id`
  Copy from input unchanged.

- `category_de`
  Copy from input unchanged.

- `category_en`
  Copy from input unchanged.

- `source`
  Copy from input unchanged.

- `title_german`
  Copy from input unchanged.

- `title_english`
  Translate the title into natural English.
  Keep the tone and rhetorical force.
  Do not over-literalize awkward phrases if a more idiomatic academic-presentable English version is clearer.

- `matched_media_mention`
  Copy from input unchanged.

- `evidence_span_german`
  Copy from input unchanged.
  For `NEUTRAL` and `IRRELEVANT`, this may be empty.

- `evidence_span_english`
  Translate the evidence span into short, telling English.
  This should be presentation-ready and should preserve the accusation.
  If the German evidence span is empty, leave this empty too.

- `excerpt_german`
  Copy from input unchanged.

- `excerpt_english`
  Translate the German excerpt into fluent English.
  Preserve the evaluative tone.
  Keep it concise and readable.
  Do not summarize it down to one sentence if the original excerpt clearly contains the key logic of the accusation.

- `why_this_was_classified_en`
  Write `1-2` short sentences in English.
  Explain why this row fits the assigned category.
  Explicitly connect the category to the wording in the excerpt and especially the evidence span.
  This should help a lecturer immediately understand the coding decision.

- `why_the_evidence_matters_en`
  Write `1` short sentence in English.
  Explain what the evidence span signals.
  Example logic:
  - for partisan-bias labels, explain that it attributes ideological or political alignment;
  - for manipulation labels, explain that it signals omission, framing, or distortion;
  - for disinformation labels, explain that it alleges falsehood or lying;
  - for incompetence labels, explain that it signals failure, weakness, or poor journalism;
  - for neutral and irrelevant, leave this empty if there is no evidence span.

- `lecturer_takeaway_en`
  Write `1` short sentence in English.
  This should be the “so what?” line for presentation use.
  Make it sound like a compact interpretation someone could read aloud in class.

Style constraints:

- Be concrete, not abstract.
- Avoid jargon unless it is necessary.
- Keep each explanation short enough to fit comfortably in a showcase table.
- Do not moralize.
- Do not argue with the label; explain it faithfully from the given annotation result.
- Do not translate outlet names unless they are already standard in English.

Special handling for `NEUTRAL`:

- Make clear that the media actor is only cited or referenced.
- Do not invent a criticism where none is present.
- `why_this_was_classified_en` should explain that the mention functions as a source reference or descriptive mention rather than an accusation.

Special handling for `IRRELEVANT`:

- Make clear that the matched term is not actually functioning as a media reference in context.
- Do not force a media interpretation.
- If the matched term refers to something else literally, explain that briefly.

Quality bar:

- A lecturer who has never seen the German data should be able to read one row and understand:
  - what media actor was mentioned,
  - what the accusation or non-accusation is,
  - what phrase triggered the label,
  - why that label makes sense.

Return the final CSV only.

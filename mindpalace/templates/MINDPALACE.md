---
schema_version: 1
entity_types: [person, concept, paper, project, term, theme]
# An edge type may also declare `domain: [entity types]` and, if directed,
# `range: [entity types]`. A proposal whose endpoints violate the signature is
# swapped when the other direction fits (and marked direction_corrected), or
# loses its type but keeps its endpoints. Unconstrained by default.
edge_types:
  relates-to: {directed: false, cluster_weight: 1.0}
  contradicts: {directed: false, cluster_weight: 1.0}
  supports: {directed: true, cluster_weight: 1.0}
  example-of: {directed: true, cluster_weight: 1.0}
  part-of: {directed: true, cluster_weight: 1.0}
  derived-from: {directed: true, cluster_weight: 1.0}
  mentions: {directed: true, cluster_weight: 0.5}
thresholds:
  cluster_activation_entities: 150
  abstain_bm25_floor: 2.0
  abstain_cosine_floor: 0.35
  community_lineage_jaccard: 0.5
  # Optional: two entities whose profiles embed closer than this are flagged
  # as `similar_entity` for merge_entities. Defaults to 0.92 if omitted.
  duplicate_cosine_floor: 0.92
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Read any nearest notes you need, then call write_note with this capture id. The
note body must contain your analysis — claims, why it matters — not a
restatement of the capture. Extract entities with a type and a one-line
description. Propose a relationship only where you can give a specific rationale
citing both endpoints; give each a strength 1-10. Reuse an existing entity name
over inventing a near-duplicate. Check previously_dismissed before re-proposing a
pair, and only re-propose when your evidence genuinely differs.
If no configured type fits an entity or relationship, give the type you mean in
plain words rather than forcing the nearest configured one; it is kept as a
proposal for the user to adopt. When a claim has a date, give valid_from and, if
it ended, valid_to (YYYY, YYYY-MM or YYYY-MM-DD, or "unknown" for ended-but-
unknown-when); never put the capture's own date in valid_to. When a claim
corrects an earlier one, name it in supersedes. Afterwards, tell the user the
`landed` line the tool returns, not what you meant to record.
Proposing nothing is a valid outcome.

## template: report_next
Write one report per community using write_community_report. Ground every finding
in the supplied members with an inline citation of the form
[Data: Entities (e_slug); Assertions (x_id)], and list the same ids in cites.
Do not state anything the supplied members do not support.

## template: abstain
Nothing in the vault is relevant to this query. Say so rather than answering from
your own knowledge.

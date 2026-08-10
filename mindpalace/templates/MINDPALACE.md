---
schema_version: 1
entity_types: [person, concept, paper, project, term, theme]
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
embedder: {kind: local, model: BAAI/bge-small-en-v1.5}
---

## template: extraction_next
Read any nearest notes you need, then call write_note with this capture id. The
note body must contain your analysis — claims, why it matters — not a
restatement of the capture. Extract entities with a type and a one-line
description. Propose a relationship only where you can give a specific rationale
citing both endpoints; give each a strength 1-10. Reuse an existing entity name
over inventing a near-duplicate. Check previously_dismissed before re-proposing a
pair, and only re-propose when your evidence genuinely differs. Proposing nothing
is a valid outcome.

## template: report_next
Write one report per community using write_community_report. Ground every finding
in the supplied members with an inline citation of the form
[Data: Entities (e_slug); Assertions (x_id)], and list the same ids in cites.
Do not state anything the supplied members do not support.

## template: abstain
Nothing in the vault is relevant to this query. Say so rather than answering from
your own knowledge.

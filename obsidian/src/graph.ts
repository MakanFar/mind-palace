/** The contract with `.graph/graph.json` (docs/decisions/0003 §Part 1). */

export type Status = "proposed" | "confirmed" | "dismissed" | "superseded";

export interface GraphEntity {
  slug: string;
  type: string;
  rank: number;
  merged_from: string[];
  /** Named by an entity instance in some note, not just implied by an
   *  assertion (docs/decisions/0004). Absent in an older export: declared. */
  declared?: boolean;
  description: string;
  stale: boolean;
  note_ids: string[];
  text_unit_ids: string[];
}

export interface GraphAssertion {
  id: string;
  note_id: string;
  status: Status;
  strength: number;
  description: string;
  direction_corrected: boolean;
  text_unit_ids: string[];
}

export interface GraphEdge {
  key: string;
  source: string;
  target: string;
  type: string | null;
  proposed_type: string | null;
  directed: boolean;
  weight: number;
  traversable: boolean;
  assertions: GraphAssertion[];
}

export interface GraphUntyped {
  id: string;
  source: string;
  target: string;
  proposed_type: string | null;
  status: Status;
  note_id: string;
  description: string;
  text_unit_ids: string[];
}

export interface GraphClaim {
  id: string;
  subject: string;
  text: string;
  status: Status;
  valid_from: string | null;
  valid_to: string | null;
  supersedes: string | null;
  note_id: string;
  text_unit_ids: string[];
}

export interface GraphCommunity {
  lineage_id: string;
  level: number;
  parent: string | null;
  members: string[];
  title: string | null;
  stale: boolean | null;
}

export interface GraphVocabulary {
  kind: string;
  proposed: string;
  count: number;
  example: string;
}

export interface GraphData {
  version: number;
  generated_at: string;
  entities: GraphEntity[];
  edges: GraphEdge[];
  untyped: GraphUntyped[];
  claims: GraphClaim[];
  communities: GraphCommunity[];
  vocabulary: GraphVocabulary[];
  captures: Record<string, { path: string; title: string }>;
  notes: Record<string, { path: string }>;
  units: Record<string, { capture_id: string; locator: string | null }>;
}

export const GRAPH_VERSION = 1;

/** Whether a row still has anything to show: dismissed is drawn as nothing. */
export function isLive(item: { status: Status }): boolean {
  return item.status !== "dismissed";
}

/** An edge is live while any member assertion is. */
export function isLiveEdge(edge: GraphEdge): boolean {
  return edge.assertions.some(isLive);
}

/** An edge still has something to review while any member is proposed. */
export function hasPending(edge: GraphEdge): boolean {
  return edge.assertions.some((a) => a.status === "proposed");
}

export function parseGraph(text: string): GraphData {
  const data = JSON.parse(text) as Partial<GraphData>;
  if (data.version !== GRAPH_VERSION) {
    throw new Error(`unsupported graph.json version ${data.version}`);
  }
  return data as GraphData;
}

export function emptyGraph(): GraphData {
  return {
    version: GRAPH_VERSION,
    generated_at: "",
    entities: [],
    edges: [],
    untyped: [],
    claims: [],
    communities: [],
    vocabulary: [],
    captures: {},
    notes: {},
    units: {},
  };
}

/** Slugs named by an assertion or claim that is not dismissed: the ones
 *  that keep an entity in the graph (docs/decisions/0004 §Part 1). */
export function liveSlugs(graph: GraphData): Set<string> {
  const live = new Set<string>();
  for (const edge of graph.edges) {
    if (edge.assertions.some(isLive)) live.add(edge.source).add(edge.target);
  }
  for (const item of graph.untyped) {
    if (isLive(item)) live.add(item.source).add(item.target);
  }
  for (const claim of graph.claims) {
    if (isLive(claim)) live.add(claim.subject);
  }
  return live;
}

/** Entities with nothing live attached: what retire would accept. */
export function isolatedEntities(graph: GraphData): GraphEntity[] {
  const live = liveSlugs(graph);
  return graph.entities.filter((e) => !live.has(e.slug));
}

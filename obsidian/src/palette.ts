/** Fixed visual language (docs/decisions/0003 §Part 3). */

export const TYPE_COLOURS: Record<string, string> = {
  person: "#e07a5f",
  concept: "#3d84a8",
  paper: "#81b29a",
  project: "#f2cc8f",
  term: "#9a8c98",
  theme: "#c9ada7",
  unknown: "#8a8a8a",
};

export const OTHER_COLOUR = "#6c757d";

export function colourFor(type: string): string {
  return TYPE_COLOURS[type] ?? OTHER_COLOUR;
}

export function radiusFor(rank: number): number {
  return Math.min(18, 4 + 2 * Math.sqrt(Math.max(0, rank)));
}

export type EdgeKind = "confirmed" | "proposed" | "untyped";

export const EDGE_STYLE: Record<EdgeKind, { dash: number[]; alpha: number }> = {
  confirmed: { dash: [], alpha: 1 },
  proposed: { dash: [6, 4], alpha: 0.5 },
  untyped: { dash: [2, 4], alpha: 0.25 },
};

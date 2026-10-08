// Card kinds as the solver sees them (solver/include/sk/solver/kinds.hpp):
//   0..55 colored: suit = floor(k / 14), value = k % 14 + 1
//   56 Flucht (x5), 57 Meerjungfrau (x2), 58 Pirat (x5), 59 Tigress, 60 Skull King

export type Suit = 0 | 1 | 2 | 3;
export const SUITS: { name: string; short: string; glyph: Glyph }[] = [
  { name: "Gelb", short: "Schatztruhe", glyph: "chest" },
  { name: "Grün", short: "Papagei", glyph: "parrot" },
  { name: "Lila", short: "Schatzkarte", glyph: "map" },
  { name: "Schwarz", short: "Piratenflagge · Trumpf", glyph: "flag" },
];

export type Glyph =
  | "chest" | "parrot" | "map" | "flag"
  | "escape" | "mermaid" | "pirate" | "tigress" | "king";

export const ESCAPE = 56, MERMAID = 57, PIRATE = 58, TIGRESS = 59, SKULL_KING = 60;
export const N_KINDS = 61;
export const TIGRESS_ESCAPE = 240, TIGRESS_PIRATE = 241;

export const SPECIALS = [ESCAPE, MERMAID, PIRATE, TIGRESS, SKULL_KING];

export function isColored(k: number) {
  return k < 56;
}
export function suitOf(k: number): Suit {
  return Math.floor(k / 14) as Suit;
}
export function valueOf(k: number) {
  return (k % 14) + 1;
}
export function multiplicity(k: number) {
  return k === ESCAPE || k === PIRATE ? 5 : k === MERMAID ? 2 : 1;
}

const SPECIAL_INFO: Record<number, { name: string; glyph: Glyph }> = {
  [ESCAPE]: { name: "Flucht", glyph: "escape" },
  [MERMAID]: { name: "Meerjungfrau", glyph: "mermaid" },
  [PIRATE]: { name: "Pirat", glyph: "pirate" },
  [TIGRESS]: { name: "Tigress", glyph: "tigress" },
  [SKULL_KING]: { name: "Skull King", glyph: "king" },
};

export function glyphOf(k: number): Glyph {
  return isColored(k) ? SUITS[suitOf(k)].glyph : SPECIAL_INFO[k].glyph;
}

export function cardName(k: number) {
  return isColored(k) ? `${SUITS[suitOf(k)].name} ${valueOf(k)}` : SPECIAL_INFO[k].name;
}

/** CSS class that colors a card: suit-0..3 or the special's name. */
export function tone(k: number) {
  return isColored(k) ? `suit-${suitOf(k)}` : `sp-${SPECIAL_INFO[k].glyph}`;
}

/** Sort for hands: specials first (strongest first), then suits, high to low. */
export function handOrder(a: number, b: number) {
  const rank = (k: number) => (isColored(k) ? 100 + suitOf(k) * 20 + (14 - valueOf(k)) : 60 - k);
  return rank(a) - rank(b);
}

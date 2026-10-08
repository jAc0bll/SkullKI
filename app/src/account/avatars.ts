// Avatars from DiceBear (https://github.com/dicebear/dicebear). An avatar is
// {style, seed, bg} plus any parts the player picked by hand (face shape,
// eyes, hair, hat, clothes ...) and colours. Everything not picked comes from
// the seed. The SVG is generated on the device.
import { createAvatar } from '@dicebear/core';
import * as adventurer from '@dicebear/adventurer';
import * as avataaars from '@dicebear/avataaars';
import * as bottts from '@dicebear/bottts';
import * as funEmoji from '@dicebear/fun-emoji';
import * as lorelei from '@dicebear/lorelei';
import * as notionists from '@dicebear/notionists';
import * as pixelArt from '@dicebear/pixel-art';

export interface AvatarSpec {
  style: StyleId;
  seed: string;
  bg?: string; // hex without '#'
  /** part -> chosen variant; null = "ohne" (only for optional parts) */
  parts?: Record<string, string | null>;
  /** colour option (e.g. skinColor) -> hex without '#' */
  colors?: Record<string, string>;
}

export const STYLES = [
  { id: 'avataaars', label: 'Comic', mod: avataaars, credit: '„Avataaars“ von Pablo Stanley' },
  { id: 'adventurer', label: 'Abenteuer', mod: adventurer, credit: '„Adventurer“ von Lisa Wischofsky, CC BY 4.0' },
  { id: 'lorelei', label: 'Lorelei', mod: lorelei, credit: '„Lorelei“ von Lisa Wischofsky, CC0' },
  { id: 'notionists', label: 'Skizze', mod: notionists, credit: '„Notionists“ von Zoish, CC0' },
  { id: 'pixel-art', label: 'Pixel', mod: pixelArt, credit: '„Pixel Art“ von DiceBear, CC0' },
  { id: 'fun-emoji', label: 'Emoji', mod: funEmoji, credit: '„Fun Emoji“ von Davis Uche, CC BY 4.0' },
  { id: 'bottts', label: 'Roboter', mod: bottts, credit: '„Bottts“ von Pablo Stanley' },
] as const;
export type StyleId = (typeof STYLES)[number]['id'];

export const BACKGROUNDS = ['b6e3f4', 'c0aede', 'd1d4f9', 'ffd5dc', 'ffdfbf', 'c1f4c5', 'fde68a', '1c1c1e', 'transparent'];

// ---- what can be edited per style (read from the style's own schema) ------

const PART_LABEL: Record<string, string> = {
  head: 'Kopfform', face: 'Gesicht', eyes: 'Augen', eyebrows: 'Augenbrauen', brows: 'Augenbrauen', nose: 'Nase',
  mouth: 'Mund', lips: 'Mund', hair: 'Frisur', top: 'Frisur & Hut', hat: 'Hut', beard: 'Bart', facialHair: 'Bart',
  glasses: 'Brille', accessories: 'Accessoires', earrings: 'Ohrringe', features: 'Merkmale', freckles: 'Sommersprossen',
  clothing: 'Kleidung', clothingGraphic: 'Motiv', body: 'Oberkörper', bodyIcon: 'Anstecker', gesture: 'Geste',
  hairAccessories: 'Haarschmuck', sides: 'Seiten', texture: 'Muster',
};
const PART_ORDER = Object.keys(PART_LABEL);
const SKIP = new Set(['style', 'base']);

const COLOR_LABEL: Record<string, string> = {
  skinColor: 'Haut', hairColor: 'Haare', eyesColor: 'Augen', eyebrowsColor: 'Augenbrauen', facialHairColor: 'Bart',
  clothesColor: 'Kleidung', clothingColor: 'Kleidung', hatColor: 'Hut', accessoriesColor: 'Accessoires',
  glassesColor: 'Brille', mouthColor: 'Mund', noseColor: 'Nase', earringsColor: 'Ohrringe', frecklesColor: 'Sommersprossen',
  hairAccessoriesColor: 'Haarschmuck', baseColor: 'Farbe',
};
// used when a style offers fewer than 4 suggestions itself
const SKIN = ['ffdbb4', 'edb98a', 'd08b5b', 'ae5d29', '614335', 'f8d25c', 'fd9841'];
const HAIR = ['2c1b18', '4a312c', '724133', 'a55728', 'b58143', 'd6b370', 'e8e1e1', 'c93305', 'f59797', '65c9ff'];
const ANY = ['262e33', '65c9ff', '5199e4', '25557c', 'e6e6e6', '929598', '3c4f5c', 'b1e2ff', 'a7ffc4', 'ffafb9', 'ffffb1', 'ff488e', 'ff5c5c', 'ffffff'];

export interface PartDef {
  key: string;
  label: string;
  variants: string[];
  optional: boolean; // has "ohne"
}
export interface ColorDef {
  key: string;
  label: string;
  palette: string[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Schema = { properties: Record<string, any> };
const styleOf = (id: StyleId) => STYLES.find((s) => s.id === id) ?? STYLES[0];
const propsOf = (id: StyleId) => (styleOf(id).mod as unknown as { schema: Schema }).schema.properties;

export function editorModel(id: StyleId): { parts: PartDef[]; colors: ColorDef[] } {
  const p = propsOf(id);
  const parts = Object.entries(p)
    .filter(([k, v]) => !SKIP.has(k) && v.type === 'array' && v.items?.enum?.length > 1)
    .map(([k, v]) => ({ key: k, label: PART_LABEL[k] ?? k, variants: v.items.enum as string[], optional: !!p[`${k}Probability`] }))
    .sort((a, b) => idx(a.key) - idx(b.key));
  const colors = Object.entries(p)
    .filter(([k]) => /Color$/.test(k) && k !== 'backgroundColor')
    .map(([k, v]) => {
      const own = ((v.default ?? []) as string[]).filter((c) => c !== 'transparent');
      const fallback = k === 'skinColor' ? SKIN : /hair|eyebrows|facialHair|beard/i.test(k) ? HAIR : ANY;
      return { key: k, label: COLOR_LABEL[k] ?? k, palette: [...new Set(own.length >= 4 ? own : [...own, ...fallback])] };
    })
    .sort((a, b) => (a.key === 'skinColor' ? -1 : b.key === 'skinColor' ? 1 : 0));
  return { parts, colors };
}
const idx = (k: string) => (PART_ORDER.includes(k) ? PART_ORDER.indexOf(k) : 99);

// ---- rendering --------------------------------------------------------------

const cache = new Map<string, string>();

/** Close-up for small face parts in the editor. */
export interface Zoom {
  scale: number; // percent
  ty: number; // percent of the size, + = down
}
const FACE_PARTS = new Set(['eyes', 'eyebrows', 'brows', 'nose', 'mouth', 'lips', 'glasses', 'earrings', 'features', 'freckles', 'beard', 'facialHair']);
export const zoomFor = (part: string): Zoom | undefined => (FACE_PARTS.has(part) ? { scale: 165, ty: 6 } : undefined);

export function avatarSvg(a: AvatarSpec, zoom?: Zoom): string {
  const key = JSON.stringify(a) + (zoom ? `|${zoom.scale}|${zoom.ty}` : '');
  let svg = cache.get(key);
  if (!svg) {
    const props = propsOf(a.style);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const opts: Record<string, any> = {
      seed: a.seed,
      radius: 50,
      backgroundColor: a.bg ? [a.bg] : undefined,
      ...(zoom ? { scale: zoom.scale, translateY: zoom.ty } : {}),
    };
    for (const [part, v] of Object.entries(a.parts ?? {})) {
      if (!props[part]) continue;
      const prob = props[`${part}Probability`] ? `${part}Probability` : null;
      if (v === null) {
        if (prob) opts[prob] = 0;
      } else {
        opts[part] = [v];
        if (prob) opts[prob] = 100;
      }
    }
    for (const [c, hex] of Object.entries(a.colors ?? {})) if (props[c]) opts[c] = [hex];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    svg = createAvatar(styleOf(a.style).mod as any, opts)
      .toString()
      .replace(/<metadata[\s\S]*?<\/metadata>/, ''); // license info, kept in STYLES.credit
    cache.set(key, svg);
    if (cache.size > 600) cache.delete(cache.keys().next().value!);
  }
  return svg;
}

export const randomSeed = () => Math.random().toString(36).slice(2, 10);

export function randomAvatar(): AvatarSpec {
  const style = STYLES[Math.floor(Math.random() * (STYLES.length - 1))].id; // robots are for the bots
  return { style, seed: randomSeed(), bg: BACKGROUNDS[Math.floor(Math.random() * 7)] };
}

/** Bots look like robots. */
export const botAvatar = (name: string): AvatarSpec => ({ style: 'bottts', seed: name, bg: '1c1c1e' });

// Avatars from DiceBear (https://github.com/dicebear/dicebear): an avatar is
// just {style, seed, bg}; the SVG is generated on the device.
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
}

export const STYLES = [
  { id: 'adventurer', label: 'Abenteuer', mod: adventurer, credit: '„Adventurer“ von Lisa Wischofsky, CC BY 4.0' },
  { id: 'avataaars', label: 'Comic', mod: avataaars, credit: '„Avataaars“ von Pablo Stanley' },
  { id: 'lorelei', label: 'Lorelei', mod: lorelei, credit: '„Lorelei“ von Lisa Wischofsky, CC0' },
  { id: 'notionists', label: 'Skizze', mod: notionists, credit: '„Notionists“ von Zoish, CC0' },
  { id: 'pixel-art', label: 'Pixel', mod: pixelArt, credit: '„Pixel Art“ von DiceBear, CC0' },
  { id: 'fun-emoji', label: 'Emoji', mod: funEmoji, credit: '„Fun Emoji“ von Davis Uche, CC BY 4.0' },
  { id: 'bottts', label: 'Roboter', mod: bottts, credit: '„Bottts“ von Pablo Stanley' },
] as const;
export type StyleId = (typeof STYLES)[number]['id'];

export const BACKGROUNDS = ['b6e3f4', 'c0aede', 'd1d4f9', 'ffd5dc', 'ffdfbf', 'c1f4c5', 'fde68a', '1c1c1e'];

const cache = new Map<string, string>();

export function avatarSvg(a: AvatarSpec): string {
  const key = `${a.style}|${a.seed}|${a.bg ?? ''}`;
  let svg = cache.get(key);
  if (!svg) {
    const style = STYLES.find((s) => s.id === a.style) ?? STYLES[0];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    svg = createAvatar(style.mod as any, {
      seed: a.seed,
      backgroundColor: a.bg ? [a.bg] : undefined,
      radius: 50,
    })
      .toString()
      .replace(/<metadata[\s\S]*?<\/metadata>/, ''); // license info, kept in STYLES.credit
    cache.set(key, svg);
    if (cache.size > 300) cache.delete(cache.keys().next().value!);
  }
  return svg;
}

export const randomSeed = () => Math.random().toString(36).slice(2, 10);

export function randomAvatar(): AvatarSpec {
  const style = STYLES[Math.floor(Math.random() * (STYLES.length - 1))].id; // robots are for the bots
  return { style, seed: randomSeed(), bg: BACKGROUNDS[Math.floor(Math.random() * (BACKGROUNDS.length - 1))] };
}

/** Bots look like robots. */
export const botAvatar = (name: string): AvatarSpec => ({ style: 'bottts', seed: name, bg: '1c1c1e' });

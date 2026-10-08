// Dark, Apple-style palette. Content (cards, table) is vivid; the controls
// float above it as Liquid Glass (see Glass.tsx).
import { Platform } from 'react-native';

export const C = {
  bg: '#05060A',
  text: '#F5F5F7',
  text2: 'rgba(235,235,245,0.62)',
  text3: 'rgba(235,235,245,0.32)',
  hairline: 'rgba(255,255,255,0.10)',
  surface: 'rgba(255,255,255,0.055)',
  surface2: 'rgba(255,255,255,0.09)',
  accent: '#5AC8FA',
  accentDeep: '#0A84FF',
  good: '#30D158',
  bad: '#FF453A',
  gold: '#FFD60A',
};

export const R = {
  card: 12,
  chip: 999,
  section: 28, // concentric with the screen corners
  inner: 18,
};

export const S = { xs: 4, s: 8, m: 12, l: 16, xl: 24, xxl: 32 };

export const font = {
  // SF Pro on Apple platforms; system UI font elsewhere.
  family: Platform.select({ web: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", system-ui, sans-serif', default: undefined }),
  tabular: { fontVariant: ['tabular-nums' as const] },
};

export const type = {
  largeTitle: { fontSize: 34, fontWeight: '800' as const, letterSpacing: -0.6, color: C.text, fontFamily: font.family },
  title: { fontSize: 22, fontWeight: '700' as const, letterSpacing: -0.3, color: C.text, fontFamily: font.family },
  headline: { fontSize: 17, fontWeight: '600' as const, color: C.text, fontFamily: font.family },
  body: { fontSize: 16, fontWeight: '400' as const, color: C.text, fontFamily: font.family },
  callout: { fontSize: 15, fontWeight: '500' as const, color: C.text2, fontFamily: font.family },
  footnote: { fontSize: 13, fontWeight: '500' as const, color: C.text2, fontFamily: font.family },
  caption: { fontSize: 12, fontWeight: '600' as const, color: C.text3, fontFamily: font.family },
};

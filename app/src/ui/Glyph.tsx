// Line icons for suits and characters, drawn for this app (24x24, stroke).
import type { ReactElement } from 'react';
import { View } from 'react-native';
import Svg, { Circle, Path, Rect } from 'react-native-svg';
import type { Glyph as G } from '@/solver/cards';

const art = (c: string): Record<G, ReactElement> => ({
  chest: (
    <>
      <Path d="M4 11a8 5.5 0 0 1 16 0" />
      <Rect x="4" y="11" width="16" height="8.5" rx="1.4" />
      <Path d="M4 14.5h16" />
      <Rect x="10.4" y="12.8" width="3.2" height="3.6" rx=".6" fill={c} stroke="none" />
    </>
  ),
  parrot: (
    <>
      <Path d="M15.5 4.5c-4 0-7 3-7 7.5 0 2.6 1 4.8 2.6 6.5" />
      <Path d="M15.5 4.5c2.3 0 4 1.6 4 3.7 0 1-.5 1.8-1.4 2.3l-2.6-1.1" />
      <Path d="M8.5 12c-1.8 1.4-3 3.8-3 7.5 2.4-.3 4.3-1.2 5.6-1" />
      <Circle cx="15.6" cy="7.4" r=".95" fill={c} stroke="none" />
    </>
  ),
  map: (
    <>
      <Path d="M3.5 6.5 9 4.5l6 2 5.5-2v13l-5.5 2-6-2-5.5 2z" />
      <Path d="M9 4.5v13M15 6.5v13" strokeOpacity={0.55} />
      <Path d="m11 10.5 2 2m0-2-2 2" />
    </>
  ),
  flag: (
    <>
      <Path d="M5.5 21V3.5" />
      <Path d="M5.5 4h13l-2.6 4.2 2.6 4.3h-13" />
      <Circle cx="11" cy="8.2" r="1.7" />
    </>
  ),
  escape: (
    <>
      <Path d="M6 21V3.5" />
      <Path d="M6 4c3-1.6 5.5 1.6 9 0s3.5-.4 3.5-.4v8.6s-1.2-1-3.5 0-6 1.6-9 0" />
    </>
  ),
  mermaid: (
    <>
      <Path d="M12 3.5c-1.5 3.5-1 6.8 0 9.5s.8 4.3-1.2 6" />
      <Path d="M10.8 19c-1.6-.4-3.8.2-5.3 1.5 2.1.6 4.5.3 5.3-1.5Z" />
      <Path d="M10.8 19c.9 1.6 3.1 2.3 5.3 1.9-1.2-1.5-3.5-2.3-5.3-1.9Z" />
    </>
  ),
  pirate: (
    <>
      <Path d="M5 4.5 17.5 17M19 4.5 6.5 17" />
      <Path d="m15 19.5 4.5-4.5M9 19.5 4.5 15" />
    </>
  ),
  tigress: (
    <>
      <Path d="M7 4c1 5.5.6 10.5-2 16" />
      <Path d="M12 3.5c.9 5.8.5 11-1.2 16.5" />
      <Path d="M17 4c.6 5.5 0 10.5-2.6 16" />
    </>
  ),
  king: (
    <>
      <Path d="M3.5 8.5 7.8 12l4.2-6.5 4.2 6.5 4.3-3.5-1.8 10H5.3z" />
      <Path d="M5.3 18.5h13.4" />
      <Circle cx="12" cy="4" r="1" fill={c} stroke="none" />
    </>
  ),
});

export function Glyph({ name, size = 18, color = '#fff', weight = 1.7 }: { name: G; size?: number; color?: string; weight?: number }) {
  // Wrapped in a View so it stacks like other views (on web a bare <svg>
  // would paint below absolutely positioned siblings such as glass layers).
  return (
    <View style={{ width: size, height: size }}>
      <Svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={weight} strokeLinecap="round" strokeLinejoin="round">
        {art(color)[name]}
      </Svg>
    </View>
  );
}

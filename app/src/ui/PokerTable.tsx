// A card table: leather rim, green felt, and the current trick as a pile in
// the middle. Cards fly in from the player who laid them and land slightly
// rotated on top of each other.
import { LinearGradient } from 'expo-linear-gradient';
import { useEffect, type ReactNode } from 'react';
import { StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withDelay, withSpring, withTiming } from 'react-native-reanimated';
import Svg, { Defs, Ellipse, RadialGradient, Stop } from 'react-native-svg';
import { Glyph } from './Glyph';
import { PlayingCard } from './PlayingCard';
import { C, font } from './theme';

/** Oval table filling its container; children are drawn on top. */
export function PokerTable({ width, height, children, style }: { width: number; height: number; children?: ReactNode; style?: StyleProp<ViewStyle> }) {
  const inset = { x: 10, y: 26 };
  const w = width - 2 * inset.x;
  const h = height - 2 * inset.y;
  return (
    <View style={[{ width, height }, style]}>
      {/* rim */}
      <View style={[styles.rim, { left: inset.x, top: inset.y, width: w, height: h, borderRadius: Math.min(w, h) / 2 }]}>
        <LinearGradient colors={['#4a2e1c', '#2a1810', '#170c07']} style={[StyleSheet.absoluteFill, { borderRadius: Math.min(w, h) / 2 }]} />
        <LinearGradient
          colors={['rgba(255,255,255,0.22)', 'rgba(255,255,255,0)']}
          locations={[0, 0.25]}
          style={[StyleSheet.absoluteFill, { borderRadius: Math.min(w, h) / 2 }]}
        />
      </View>
      {/* felt */}
      <View style={{ position: 'absolute', left: inset.x + 12, top: inset.y + 12 }} pointerEvents="none">
        <Svg width={w - 24} height={h - 24}>
          <Defs>
            <RadialGradient id="felt" cx="50%" cy="45%" rx="60%" ry="60%">
              <Stop offset="0" stopColor="#23875a" />
              <Stop offset="0.6" stopColor="#156343" />
              <Stop offset="1" stopColor="#0a3423" />
            </RadialGradient>
          </Defs>
          <Ellipse cx={(w - 24) / 2} cy={(h - 24) / 2} rx={(w - 24) / 2} ry={(h - 24) / 2} fill="url(#felt)" />
          <Ellipse
            cx={(w - 24) / 2}
            cy={(h - 24) / 2}
            rx={(w - 24) / 2 - 12}
            ry={(h - 24) / 2 - 12}
            fill="none"
            stroke="rgba(255,214,10,0.28)"
            strokeWidth={1.2}
            strokeDasharray="2 6"
          />
        </Svg>
      </View>
      {/* watermark */}
      <View style={[StyleSheet.absoluteFill, styles.center]} pointerEvents="none">
        <View style={{ opacity: 0.1 }}>
          <Glyph name="king" size={Math.min(w, h) * 0.32} color="#ffffff" weight={1.2} />
        </View>
      </View>
      {children}
    </View>
  );
}

// Direction towards each seat, relative to me: bottom, left, top, right.
const DIR = [
  { x: 0, y: 1 },
  { x: -1, y: 0 },
  { x: 0, y: -1 },
  { x: 1, y: 0 },
];

export interface PileEntry {
  kind: number;
  /** seat relative to me: 0 bottom (me), 1 left, 2 top, 3 right */
  rel: number;
  label?: string;
}

/** The trick in the middle of the table. `winner` = index of the winning entry once complete. */
export function TrickPile({ cards, winner, caption, size = 'lg' }: { cards: PileEntry[]; winner?: number; caption?: string; size?: 'lg' | 'md' }) {
  return (
    <View style={[StyleSheet.absoluteFill, styles.center]} pointerEvents="none">
      <View style={{ width: 0, height: 0 }}>
        {cards.map((c, i) => (
          <PileCard
            key={`${i}-${c.kind}-${c.rel}`}
            entry={c}
            index={i}
            size={size}
            state={winner === undefined ? 'live' : winner === i ? 'won' : 'lost'}
          />
        ))}
        {caption && (
          <View style={[styles.caption, { top: CARD[size].h / 2 + CARD[size].h * 0.22 + 10 }]}>
            <Text style={styles.captionText} numberOfLines={1}>
              {caption}
            </Text>
          </View>
        )}
      </View>
    </View>
  );
}

const CARD = { lg: { w: 66, h: 96 }, md: { w: 52, h: 76 } };

function PileCard({ entry, index, size, state }: { entry: PileEntry; index: number; size: 'lg' | 'md'; state: 'live' | 'won' | 'lost' }) {
  const d = DIR[entry.rel];
  const { w, h } = CARD[size];
  // where it lands: a little towards its player, with a stable random tilt
  const tilt = (((entry.kind * 37 + index * 53 + entry.rel * 19) % 19) - 9) * 1.1;
  const fx = d.x * w * 0.38;
  const fy = d.y * h * 0.22;
  const k = useSharedValue(0);
  const glow = useSharedValue(0);
  useEffect(() => {
    k.value = withSpring(1, { damping: 15, stiffness: 160, mass: 0.8 });
  }, [k]);
  useEffect(() => {
    glow.value = withDelay(150, withTiming(state === 'live' ? 0 : 1, { duration: 350 }));
  }, [glow, state]);
  const anim = useAnimatedStyle(() => ({
    opacity: Math.min(1, k.value * 2) * (state === 'lost' ? 1 - glow.value * 0.4 : 1),
    transform: [
      { translateX: fx + (1 - k.value) * d.x * 170 },
      { translateY: fy + (1 - k.value) * d.y * 190 },
      { rotate: `${tilt * k.value + (1 - k.value) * d.x * 30}deg` },
      { scale: 0.85 + 0.15 * k.value + (state === 'won' ? glow.value * 0.08 : 0) },
    ],
  }));
  return (
    <Animated.View style={[{ position: 'absolute', left: -w / 2, top: -h / 2, zIndex: index + (state === 'won' ? 10 : 0) }, anim]}>
      <View style={state === 'won' ? styles.won : undefined}>
        <PlayingCard kind={entry.kind} size={size} />
      </View>
      {entry.label && <Text style={styles.label}>{entry.label}</Text>}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  rim: {
    position: 'absolute',
    shadowColor: '#000',
    shadowOpacity: 0.6,
    shadowRadius: 22,
    shadowOffset: { width: 0, height: 14 },
    elevation: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,214,10,0.25)',
  },
  center: { alignItems: 'center', justifyContent: 'center' },
  won: { borderRadius: 14, shadowColor: C.gold, shadowOpacity: 1, shadowRadius: 18, shadowOffset: { width: 0, height: 0 } },
  label: { textAlign: 'center', marginTop: 3, fontSize: 10, fontWeight: '800', color: C.gold, fontFamily: font.family },
  caption: {
    position: 'absolute',
    left: -90,
    width: 180,
    alignItems: 'center',
  },
  captionText: {
    overflow: 'hidden',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.5)', fontSize: 12, fontWeight: '700', color: C.gold, fontFamily: font.family },
});

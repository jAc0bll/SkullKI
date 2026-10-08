// A card as vivid content: colour gradient, big rounded number, glyph,
// and a glossy highlight. Optional strategy gauge underneath.
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import { Platform, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring, withTiming } from 'react-native-reanimated';
import { ESCAPE, MERMAID, PIRATE, SKULL_KING, TIGRESS, cardName, glyphOf, isColored, suitOf, valueOf } from '@/solver/cards';
import { Glyph } from './Glyph';
import { C, font } from './theme';

type Look = { from: string; to: string; ink: string; rim: string };
const SUIT_LOOK: Look[] = [
  { from: '#FFE066', to: '#FF9F0A', ink: '#3D2600', rim: 'rgba(255,255,255,0.55)' }, // Gelb
  { from: '#4BE36B', to: '#1A9A3A', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.4)' }, // Grün
  { from: '#D17BFF', to: '#7B35D1', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.4)' }, // Lila
  { from: '#3A3A3F', to: '#0C0C0F', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.28)' }, // Schwarz (Trumpf)
];
const SPECIAL_LOOK: Record<number, Look> = {
  [ESCAPE]: { from: '#F2F2F7', to: '#AEAEB2', ink: '#1C1C1E', rim: 'rgba(255,255,255,0.7)' },
  [MERMAID]: { from: '#70E1FF', to: '#0A6CFF', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.45)' },
  [PIRATE]: { from: '#FF6B5E', to: '#C21D15', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.4)' },
  [TIGRESS]: { from: '#FFB340', to: '#E2540A', ink: '#FFFFFF', rim: 'rgba(255,255,255,0.45)' },
  [SKULL_KING]: { from: '#2A2208', to: '#060503', ink: C.gold, rim: 'rgba(255,214,10,0.65)' },
};
export const lookOf = (k: number) => (isColored(k) ? SUIT_LOOK[suitOf(k)] : SPECIAL_LOOK[k]);

const SIZES = {
  lg: { w: 66, h: 96, num: 24, glyph: 30, r: 13 },
  md: { w: 52, h: 76, num: 19, glyph: 24, r: 11 },
  sm: { w: 40, h: 58, num: 15, glyph: 18, r: 9 },
  xs: { w: 30, h: 43, num: 12, glyph: 13, r: 7 },
};

const pctText = (p: number) => (p >= 0.995 ? '100' : p < 0.005 ? '0' : (p * 100).toFixed(p < 0.1 ? 1 : 0));
export { pctText };

export function PlayingCard({
  kind,
  size = 'md',
  p,
  best,
  dim,
  onPress,
}: {
  kind: number;
  size?: keyof typeof SIZES;
  p?: number;
  best?: boolean;
  dim?: boolean;
  onPress?: () => void;
}) {
  const z = SIZES[size];
  const look = lookOf(kind);
  const press = useSharedValue(0);
  const anim = useAnimatedStyle(() => ({
    transform: [{ scale: 1 - press.value * 0.06 }, { translateY: best ? -6 : 0 }],
  }));
  const colored = isColored(kind);
  const label = colored ? String(valueOf(kind)) : '';

  const face = (
    <Animated.View
      style={[
        { width: z.w, height: z.h, borderRadius: z.r },
        styles.face,
        best && styles.best,
        dim && { opacity: 0.32 },
        anim,
      ]}>
      <LinearGradient colors={[look.from, look.to]} start={{ x: 0.1, y: 0 }} end={{ x: 0.9, y: 1 }} style={[StyleSheet.absoluteFill, { borderRadius: z.r }]} />
      {/* gloss */}
      <LinearGradient
        colors={['rgba(255,255,255,0.38)', 'rgba(255,255,255,0)']}
        locations={[0, 0.5]}
        style={[StyleSheet.absoluteFill, { borderRadius: z.r }]}
        pointerEvents="none"
      />
      <View pointerEvents="none" style={[StyleSheet.absoluteFill, { borderRadius: z.r, borderWidth: 1, borderColor: look.rim }]} />
      {label !== '' && (
        <Text style={[styles.num, { color: look.ink, fontSize: z.num, left: z.w * 0.12, top: z.h * 0.06 }]}>{label}</Text>
      )}
      <View style={[styles.center, colored && { top: z.h * 0.22 }]}>
        <Glyph name={glyphOf(kind)} size={colored ? z.glyph * 0.85 : z.glyph} color={look.ink} weight={size === 'xs' ? 2 : 1.8} />
      </View>
      {!colored && size === 'lg' && (
        <Text numberOfLines={1} adjustsFontSizeToFit style={[styles.title, { color: look.ink }]}>
          {cardName(kind)}
        </Text>
      )}
    </Animated.View>
  );

  return (
    <View style={{ alignItems: 'center', gap: 6 }}>
      {onPress ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={cardName(kind) + (p !== undefined ? `, ${pctText(p)} Prozent` : '')}
          onPressIn={() => (press.value = withSpring(1, { damping: 18, stiffness: 400 }))}
          onPressOut={() => (press.value = withSpring(0, { damping: 14, stiffness: 300 }))}
          onPress={() => {
            if (Platform.OS !== 'web') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
          }}>
          {face}
        </Pressable>
      ) : (
        face
      )}
      {p !== undefined && <Gauge p={p} best={!!best} width={z.w} />}
    </View>
  );
}

function Gauge({ p, best, width }: { p: number; best: boolean; width: number }) {
  const fill = useAnimatedStyle(() => ({ width: withTiming(Math.max(2, p * width), { duration: 450 }) }), [p, width]);
  return (
    <View style={[styles.gauge, { width }]}>
      <Animated.View style={[styles.gaugeFill, best && { backgroundColor: C.accent }, fill]} />
      <Text style={[styles.gaugeText, best && { color: '#001826' }]}>{pctText(p)}%</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  face: {
    shadowColor: '#000',
    shadowOpacity: 0.45,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 6 },
    elevation: 6,
  },
  best: {
    shadowColor: C.accent,
    shadowOpacity: 0.9,
    shadowRadius: 18,
    shadowOffset: { width: 0, height: 0 },
  },
  num: { position: 'absolute', fontWeight: '800', letterSpacing: -0.5, fontFamily: font.family, ...font.tabular },
  center: { position: 'absolute', left: 0, right: 0, top: 0, bottom: 0, alignItems: 'center', justifyContent: 'center' },
  title: {
    position: 'absolute',
    left: 2,
    right: 2,
    bottom: 6,
    textAlign: 'center',
    fontSize: 7.5,
    fontWeight: '800',
    letterSpacing: 0.2,
    textTransform: 'uppercase',
    fontFamily: font.family,
  },
  gauge: {
    height: 20,
    borderRadius: 10,
    backgroundColor: 'rgba(255,255,255,0.08)',
    overflow: 'hidden',
    justifyContent: 'center',
  },
  gaugeFill: { position: 'absolute', left: 0, top: 0, bottom: 0, backgroundColor: 'rgba(90,200,250,0.35)', borderRadius: 10 },
  gaugeText: { textAlign: 'center', fontSize: 11.5, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
});

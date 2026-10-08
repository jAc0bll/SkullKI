// The content layer behind everything: near-black with slowly drifting
// colour light. It gives the glass controls something to refract.
import { useEffect } from 'react';
import { StyleSheet, useWindowDimensions, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useReducedMotion,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import Svg, { Defs, RadialGradient, Rect, Stop } from 'react-native-svg';
import { C } from './theme';

const ORBS = [
  { color: '#2E5BFF', x: -0.25, y: -0.12, r: 1.0, dx: 0.12, dy: 0.06, t: 21000 },
  { color: '#8E3BFF', x: 0.55, y: 0.18, r: 0.85, dx: -0.1, dy: 0.08, t: 26000 },
  { color: '#00B3A4', x: -0.1, y: 0.62, r: 0.9, dx: 0.14, dy: -0.07, t: 24000 },
  { color: '#FF7A1A', x: 0.7, y: 0.85, r: 0.55, dx: -0.08, dy: -0.06, t: 30000 },
];

function Orb({ color, x, y, r, dx, dy, t, w, h }: (typeof ORBS)[number] & { w: number; h: number }) {
  const reduce = useReducedMotion();
  const k = useSharedValue(0);
  useEffect(() => {
    if (!reduce) k.value = withRepeat(withTiming(1, { duration: t, easing: Easing.inOut(Easing.sin) }), -1, true);
  }, [k, reduce, t]);
  const size = r * Math.max(w, h);
  const style = useAnimatedStyle(() => ({
    transform: [{ translateX: x * w + k.value * dx * w }, { translateY: y * h + k.value * dy * h }],
  }));
  const id = `g${color.slice(1)}`;
  return (
    <Animated.View style={[{ position: 'absolute', width: size, height: size }, style]} pointerEvents="none">
      <Svg width={size} height={size}>
        <Defs>
          <RadialGradient id={id} cx="50%" cy="50%" r="50%">
            <Stop offset="0" stopColor={color} stopOpacity={0.55} />
            <Stop offset="0.45" stopColor={color} stopOpacity={0.18} />
            <Stop offset="1" stopColor={color} stopOpacity={0} />
          </RadialGradient>
        </Defs>
        <Rect width={size} height={size} fill={`url(#${id})`} />
      </Svg>
    </Animated.View>
  );
}

export function Backdrop() {
  const { width, height } = useWindowDimensions();
  return (
    <View style={[StyleSheet.absoluteFill, { backgroundColor: C.bg, overflow: 'hidden' }]} pointerEvents="none">
      {ORBS.map((o) => (
        <Orb key={o.color} {...o} w={width} h={height} />
      ))}
    </View>
  );
}

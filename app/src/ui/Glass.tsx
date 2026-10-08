// Liquid Glass surface. On iOS 26+ this is Apple's real material
// (UIGlassEffect via expo-glass-effect): it refracts the content behind it,
// reacts to touch and merges with neighbours inside a <GlassGroup>.
// Elsewhere (web, older iOS) a blur + specular rim imitates it.
import { BlurView } from 'expo-blur';
import { GlassContainer, GlassView, isGlassEffectAPIAvailable, isLiquidGlassAvailable } from 'expo-glass-effect';
import { LinearGradient } from 'expo-linear-gradient';
import type { ReactNode } from 'react';
import { Platform, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

export const nativeGlass =
  Platform.OS === 'ios' && (() => {
    try {
      return isLiquidGlassAvailable() && isGlassEffectAPIAvailable();
    } catch {
      return false;
    }
  })();

export function Glass({
  children,
  style,
  radius = 999,
  interactive,
  tint,
  clear,
}: {
  children?: ReactNode;
  style?: StyleProp<ViewStyle>;
  radius?: number;
  interactive?: boolean;
  tint?: string;
  clear?: boolean;
}) {
  if (nativeGlass) {
    return (
      <GlassView
        style={[{ borderRadius: radius }, style]}
        glassEffectStyle={clear ? 'clear' : 'regular'}
        tintColor={tint}
        isInteractive={interactive}>
        {children}
      </GlassView>
    );
  }
  return (
    // zIndex 0 makes this its own stacking context; the layers sit at -1 so
    // they never cover the content (on web, inputs and svgs are not
    // positioned and would otherwise end up below the blur).
    <View style={[{ borderRadius: radius, overflow: 'hidden', zIndex: 0 }, style]}>
      <BlurView intensity={Platform.OS === 'web' ? 28 : 40} tint="dark" style={[StyleSheet.absoluteFill, BEHIND]} />
      {tint && <View style={[StyleSheet.absoluteFill, BEHIND, { backgroundColor: tint, opacity: 0.55 }]} />}
      {/* specular rim: light from above */}
      <LinearGradient
        pointerEvents="none"
        colors={['rgba(255,255,255,0.20)', 'rgba(255,255,255,0.04)', 'rgba(255,255,255,0.0)']}
        locations={[0, 0.45, 1]}
        style={[StyleSheet.absoluteFill, BEHIND]}
      />
      <View
        pointerEvents="none"
        style={[
          StyleSheet.absoluteFill,
          BEHIND,
          {
            borderRadius: radius,
            borderWidth: StyleSheet.hairlineWidth * 2,
            borderColor: 'rgba(255,255,255,0.22)',
            borderBottomColor: 'rgba(255,255,255,0.06)',
          },
        ]}
      />
      {children}
    </View>
  );
}

const BEHIND = { zIndex: -1 } as const;

/** Groups glass shapes so they melt into each other when close (iOS 26). */
export function GlassGroup({ children, spacing = 10, style }: { children: ReactNode; spacing?: number; style?: StyleProp<ViewStyle> }) {
  if (nativeGlass) return <GlassContainer spacing={spacing} style={style}>{children}</GlassContainer>;
  return <View style={style}>{children}</View>;
}

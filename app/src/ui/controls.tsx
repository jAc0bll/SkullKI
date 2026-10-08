// Glass controls: capsule buttons, segmented control with a sliding glass
// thumb, chips, stepper. All give haptic feedback on iPhone.
import * as Haptics from 'expo-haptics';
import { SymbolView, type SymbolViewProps } from 'expo-symbols';
import { useState, type ReactNode } from 'react';
import { Platform, Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native';
import Animated, { useAnimatedStyle, withSpring } from 'react-native-reanimated';
import { Glass, GlassGroup } from './Glass';
import { C, font, type } from './theme';

export const tick = () => {
  if (Platform.OS !== 'web') Haptics.selectionAsync();
};

type SF = SymbolViewProps['name'];

export function Icon({ sf, web, size = 18, color = C.text }: { sf: SF; web: string; size?: number; color?: string }) {
  return <SymbolView name={sf} size={size} tintColor={color} fallback={<Text style={{ color, fontSize: size, lineHeight: size + 2 }}>{web}</Text>} />;
}

export function GlassButton({
  label,
  icon,
  onPress,
  disabled,
  prominent,
  style,
  accessibilityLabel,
}: {
  label?: string;
  icon?: ReactNode;
  onPress: () => void;
  disabled?: boolean;
  prominent?: boolean;
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel ?? label}
      disabled={disabled}
      onPress={() => {
        tick();
        onPress();
      }}
      style={({ pressed }) => [{ opacity: disabled ? 0.35 : 1, transform: [{ scale: pressed ? 0.96 : 1 }] }, style]}>
      <Glass interactive tint={prominent ? C.accentDeep : undefined} style={[styles.button, !label && styles.iconButton]}>
        {icon}
        {label && <Text style={[styles.buttonText, prominent && { color: '#fff' }]}>{label}</Text>}
      </Glass>
    </Pressable>
  );
}

/** iOS-style segmented control; the selection is a glass droplet that slides. */
export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string; sub?: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  const [w, setW] = useState(0);
  const i = Math.max(0, options.findIndex((o) => o.value === value));
  const seg = w / options.length;
  const thumb = useAnimatedStyle(
    () => ({ width: seg - 6, transform: [{ translateX: withSpring(3 + i * seg, { damping: 18, stiffness: 220, mass: 0.7 }) }] }),
    [i, seg],
  );
  const tall = options.some((o) => o.sub);
  return (
    <Glass radius={999} style={{ padding: 0 }}>
      <View style={[styles.segment, tall && { height: 52 }]} onLayout={(e) => setW(e.nativeEvent.layout.width)}>
        {w > 0 && (
          <Animated.View style={[styles.thumb, thumb]}>
            <View style={styles.thumbFill} />
          </Animated.View>
        )}
        {options.map((o) => {
          const on = o.value === value;
          return (
            <Pressable
              key={String(o.value)}
              accessibilityRole="tab"
              accessibilityState={{ selected: on }}
              style={styles.segItem}
              onPress={() => {
                if (!on) tick();
                onChange(o.value);
              }}>
              <Text style={[styles.segText, on && { color: C.text }]}>{o.label}</Text>
              {o.sub && <Text style={[styles.segSub, on && { color: C.text2 }]}>{o.sub}</Text>}
            </Pressable>
          );
        })}
      </View>
    </Glass>
  );
}

/** A row of numbered glass chips (e.g. rounds); merge into each other on iOS 26. */
export function ChipRow({ values, value, onChange }: { values: number[]; value: number; onChange: (v: number) => void }) {
  return (
    <GlassGroup spacing={6} style={styles.chipRow}>
      {values.map((v) => {
        const on = v === value;
        return (
          <Pressable
            key={v}
            accessibilityRole="radio"
            accessibilityLabel={String(v)}
            accessibilityState={{ selected: on }}
            onPress={() => {
              if (!on) tick();
              onChange(v);
            }}
            style={({ pressed }) => ({ transform: [{ scale: pressed ? 0.92 : 1 }] })}>
            <Glass interactive tint={on ? C.accentDeep : undefined} style={styles.chip}>
              <Text style={[styles.chipText, on && { color: '#fff' }]}>{v}</Text>
            </Glass>
          </Pressable>
        );
      })}
    </GlassGroup>
  );
}

/** Number stepper; with `unknown`, one step below 0 means "not entered" (shown as ?). */
export function Stepper({ value, max, onChange, unknown }: { value: number; max: number; onChange: (v: number) => void; unknown?: boolean }) {
  const lowest = unknown ? -1 : 0;
  const btn = (d: number, glyph: string, label: string) => {
    const off = d < 0 ? value <= lowest : value >= max;
    return (
      <Pressable
        accessibilityLabel={label}
        disabled={off}
        hitSlop={6}
        onPress={() => {
          tick();
          onChange(Math.max(lowest, Math.min(max, value + d)));
        }}
        style={({ pressed }) => [styles.stepBtn, { opacity: off ? 0.3 : pressed ? 0.6 : 1 }]}>
        <Text style={styles.stepGlyph}>{glyph}</Text>
      </Pressable>
    );
  };
  return (
    <Glass interactive radius={999} style={styles.stepper}>
      {btn(-1, '−', 'weniger')}
      <Text style={styles.stepValue}>{value < 0 ? '?' : value}</Text>
      {btn(+1, '+', 'mehr')}
    </Glass>
  );
}

export function SectionTitle({ children, right }: { children: string; right?: ReactNode }) {
  return (
    <View style={styles.sectionTitle}>
      <Text style={type.title}>{children}</Text>
      {right}
    </View>
  );
}

/** Content container (not glass: Apple keeps glass for the control layer). */
export function Panel({ children, style }: { children: ReactNode; style?: StyleProp<ViewStyle> }) {
  return <View style={[styles.panel, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  button: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8, height: 44, paddingHorizontal: 18 },
  iconButton: { width: 44, paddingHorizontal: 0 },
  buttonText: { ...type.headline, fontSize: 16 },
  segment: { flexDirection: 'row', height: 40 },
  thumb: { position: 'absolute', top: 3, bottom: 3, left: 0 },
  thumbFill: { flex: 1, borderRadius: 999, backgroundColor: 'rgba(255,255,255,0.16)', borderWidth: StyleSheet.hairlineWidth, borderColor: 'rgba(255,255,255,0.25)' },
  segItem: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  segText: { fontSize: 14.5, fontWeight: '600', color: C.text2, fontFamily: font.family },
  segSub: { fontSize: 10.5, fontWeight: '500', color: C.text3, marginTop: 1, fontFamily: font.family },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { width: 44, height: 44, alignItems: 'center', justifyContent: 'center' },
  chipText: { fontSize: 17, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
  stepper: { flexDirection: 'row', alignItems: 'center', height: 40 },
  stepBtn: { width: 40, height: 40, alignItems: 'center', justifyContent: 'center' },
  stepGlyph: { fontSize: 22, fontWeight: '500', color: C.text, marginTop: -2 },
  stepValue: { minWidth: 26, textAlign: 'center', fontSize: 18, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
  sectionTitle: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  panel: {
    borderRadius: 28,
    padding: 18,
    backgroundColor: 'rgba(22,24,32,0.62)',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: C.hairline,
  },
});

// Avatar generator: pick a style, roll the dice, choose from variants and a
// background colour.
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { ZoomIn } from 'react-native-reanimated';
import { GlassButton, Icon, tick } from '@/ui/controls';
import { C, font } from '@/ui/theme';
import { Avatar } from './Avatar';
import { BACKGROUNDS, STYLES, randomSeed, type AvatarSpec } from './avatars';


export function AvatarGenerator({ value, onChange }: { value: AvatarSpec; onChange: (a: AvatarSpec) => void }) {
  const [batch, setBatch] = useState(0);
  const variants = useMemo(
    () => Array.from({ length: 8 }, () => randomSeed()),
    // new variants when the style changes or on "more"
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [value.style, batch],
  );
  const set = (patch: Partial<AvatarSpec>) => {
    tick();
    onChange({ ...value, ...patch });
  };

  return (
    <View style={{ gap: 16 }}>
      <View style={styles.previewRow}>
        <Animated.View key={`${value.style}-${value.seed}-${value.bg}`} entering={ZoomIn.springify().damping(14)}>
          <Avatar avatar={value} size={112} ring="rgba(255,255,255,0.35)" />
        </Animated.View>
        <GlassButton label="Würfeln" icon={<Icon sf="dice.fill" web="🎲" size={16} />} onPress={() => set({ seed: randomSeed() })} />
      </View>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 10, paddingVertical: 2 }}>
        {STYLES.map((s) => {
          const on = s.id === value.style;
          return (
            <Pressable key={s.id} accessibilityLabel={`Stil ${s.label}`} onPress={() => set({ style: s.id })} style={[styles.styleChip, on && styles.styleOn]}>
              <Avatar avatar={{ style: s.id, seed: value.seed, bg: value.bg }} size={44} />
              <Text style={[styles.styleLabel, on && { color: C.text }]}>{s.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>

      <View>
        <View style={styles.variantHead}>
          <Text style={styles.label}>Varianten</Text>
          <Pressable onPress={() => setBatch((b) => b + 1)} hitSlop={8}>
            <Text style={[styles.label, { color: C.accent }]}>Neue zeigen</Text>
          </Pressable>
        </View>
        <View style={styles.grid}>
          {variants.map((seed) => (
            <Pressable key={seed} accessibilityLabel="Diese Variante wählen" onPress={() => set({ seed })} style={({ pressed }) => ({ transform: [{ scale: pressed ? 0.9 : 1 }] })}>
              <Avatar avatar={{ ...value, seed }} size={64} ring={seed === value.seed ? C.accent : undefined} />
            </Pressable>
          ))}
        </View>
      </View>

      <View>
        <Text style={[styles.label, { marginBottom: 8 }]}>Hintergrund</Text>
        <View style={styles.colors}>
          {BACKGROUNDS.map((bg) => (
            <Pressable
              key={bg}
              accessibilityLabel={`Hintergrund ${bg}`}
              onPress={() => set({ bg })}
              style={[styles.swatch, { backgroundColor: `#${bg}` }, value.bg === bg && styles.swatchOn]}
            />
          ))}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  previewRow: { flexDirection: 'row', alignItems: 'center', gap: 18 },
  styleChip: { alignItems: 'center', gap: 4, padding: 6, borderRadius: 16, borderWidth: 1.5, borderColor: 'transparent' },
  styleOn: { borderColor: C.accent, backgroundColor: 'rgba(90,200,250,0.10)' },
  styleLabel: { fontSize: 11, fontWeight: '600', color: C.text2, fontFamily: font.family },
  variantHead: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  label: { fontSize: 13, fontWeight: '600', color: C.text2, fontFamily: font.family },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  colors: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  swatch: { width: 34, height: 34, borderRadius: 17, borderWidth: 2, borderColor: 'rgba(255,255,255,0.15)' },
  swatchOn: { borderColor: '#fff', transform: [{ scale: 1.12 }] },
});

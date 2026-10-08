// Character editor: style, then every part (face shape, eyes, brows, nose,
// mouth, hair, beard, glasses, hat, clothes ...), colours and background.
// Built from each DiceBear style's own schema, so every style shows exactly
// what it supports.
import { useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, ZoomIn } from 'react-native-reanimated';
import { GlassButton, Icon, tick } from '@/ui/controls';
import { C, font } from '@/ui/theme';
import { Avatar } from './Avatar';
import { BACKGROUNDS, STYLES, editorModel, randomSeed, zoomFor, type AvatarSpec, type Zoom } from './avatars';

type Tab = { kind: 'part'; key: string } | { kind: 'colors' } | { kind: 'bg' };

export function AvatarGenerator({ value, onChange }: { value: AvatarSpec; onChange: (a: AvatarSpec) => void }) {
  const model = useMemo(() => editorModel(value.style), [value.style]);
  const [tab, setTab] = useState<Tab>({ kind: 'part', key: model.parts[0]?.key ?? '' });
  const activePart = tab.kind === 'part' ? model.parts.find((p) => p.key === tab.key) ?? model.parts[0] : undefined;

  const set = (a: AvatarSpec) => {
    tick();
    onChange(a);
  };
  const setPart = (key: string, v: string | null) => set({ ...value, parts: { ...value.parts, [key]: v } });
  const setColor = (key: string, hex: string) => set({ ...value, colors: { ...value.colors, [key]: hex } });
  const tabs: { id: string; label: string; tab: Tab }[] = [
    ...model.parts.map((p) => ({ id: p.key, label: p.label, tab: { kind: 'part', key: p.key } as Tab })),
    ...(model.colors.length ? [{ id: '#colors', label: 'Farben', tab: { kind: 'colors' } as Tab }] : []),
    { id: '#bg', label: 'Hintergrund', tab: { kind: 'bg' } as Tab },
  ];
  const tabId = tab.kind === 'part' ? activePart?.key : tab.kind === 'colors' ? '#colors' : '#bg';

  return (
    <View style={{ gap: 16 }}>
      <View style={styles.previewRow}>
        <Animated.View key={JSON.stringify(value)} entering={ZoomIn.springify().damping(15)}>
          <Avatar avatar={value} size={116} ring="rgba(255,255,255,0.35)" />
        </Animated.View>
        <View style={{ gap: 10, flex: 1 }}>
          <GlassButton
            label="Würfeln"
            icon={<Icon sf="dice.fill" web="🎲" size={16} />}
            onPress={() => set({ style: value.style, seed: randomSeed(), bg: value.bg })}
          />
          <GlassButton
            label="Teile zurücksetzen"
            disabled={!value.parts && !value.colors}
            onPress={() => set({ style: value.style, seed: value.seed, bg: value.bg })}
          />
        </View>
      </View>

      {/* style */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 8, paddingVertical: 2 }}>
        {STYLES.map((s) => {
          const on = s.id === value.style;
          return (
            <Pressable
              key={s.id}
              accessibilityLabel={`Stil ${s.label}`}
              onPress={() => {
                // parts differ between styles: start the new style fresh
                set({ style: s.id, seed: value.seed, bg: value.bg });
                setTab({ kind: 'part', key: editorModel(s.id).parts[0]?.key ?? '' });
              }}
              style={[styles.styleChip, on && styles.on]}>
              <Avatar avatar={{ style: s.id, seed: value.seed, bg: value.bg }} size={42} />
              <Text style={[styles.small, on && { color: C.text }]}>{s.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>

      {/* categories */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 6 }}>
        {tabs.map((t) => {
          const on = t.id === tabId;
          return (
            <Pressable
              key={t.id}
              onPress={() => {
                tick();
                setTab(t.tab);
              }}
              style={[styles.tab, on && styles.tabOn]}>
              <Text style={[styles.tabText, on && { color: '#fff' }]}>{t.label}</Text>
            </Pressable>
          );
        })}
      </ScrollView>

      {/* choices */}
      <Animated.View key={tabId} entering={FadeIn.duration(180)}>
        {activePart && (
          <View style={styles.grid}>
            {activePart.optional && (
              <Tile
                avatar={{ ...value, parts: { ...value.parts, [activePart.key]: null } }}
                on={value.parts?.[activePart.key] === null}
                zoom={zoomFor(activePart.key)}
                label="ohne"
                onPress={() => setPart(activePart.key, null)}
              />
            )}
            {activePart.variants.map((v) => (
              <Tile
                key={v}
                avatar={{ ...value, parts: { ...value.parts, [activePart.key]: v } }}
                on={value.parts?.[activePart.key] === v}
                zoom={zoomFor(activePart.key)}
                onPress={() => setPart(activePart.key, v)}
              />
            ))}
          </View>
        )}
        {tab.kind === 'colors' && (
          <View style={{ gap: 14 }}>
            {model.colors.map((c) => (
              <View key={c.key} style={{ gap: 6 }}>
                <Text style={styles.label}>{c.label}</Text>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={{ gap: 8, paddingVertical: 3 }}>
                  {c.palette.map((hex) => (
                    <Swatch key={hex} hex={hex} on={value.colors?.[c.key] === hex} onPress={() => setColor(c.key, hex)} />
                  ))}
                </ScrollView>
              </View>
            ))}
          </View>
        )}
        {tab.kind === 'bg' && (
          <View style={styles.colors}>
            {BACKGROUNDS.map((bg) => (
              <Swatch key={bg} hex={bg} on={value.bg === bg} onPress={() => set({ ...value, bg })} />
            ))}
          </View>
        )}
      </Animated.View>
    </View>
  );
}

function Tile({ avatar, on, label, zoom, onPress }: { avatar: AvatarSpec; on: boolean; label?: string; zoom?: Zoom; onPress: () => void }) {
  return (
    <Pressable
      accessibilityLabel={label ?? 'Variante wählen'}
      onPress={onPress}
      style={({ pressed }) => [styles.tile, on && styles.on, { transform: [{ scale: pressed ? 0.92 : 1 }] }]}>
      <Avatar avatar={avatar} size={58} zoom={zoom} />
      {label && <Text style={[styles.small, styles.tileLabel]}>{label}</Text>}
    </Pressable>
  );
}

function Swatch({ hex, on, onPress }: { hex: string; on: boolean; onPress: () => void }) {
  const transparent = hex === 'transparent';
  return (
    <Pressable
      accessibilityLabel={transparent ? 'Kein Hintergrund' : `Farbe ${hex}`}
      onPress={onPress}
      style={[styles.swatch, { backgroundColor: transparent ? 'transparent' : `#${hex}` }, on && styles.swatchOn]}>
      {transparent && <Text style={[styles.small, { color: C.text2 }]}>ohne</Text>}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  previewRow: { flexDirection: 'row', alignItems: 'center', gap: 18 },
  styleChip: { alignItems: 'center', gap: 4, padding: 6, borderRadius: 16, borderWidth: 1.5, borderColor: 'transparent' },
  on: { borderColor: C.accent, backgroundColor: 'rgba(90,200,250,0.12)' },
  small: { fontSize: 11, fontWeight: '600', color: C.text2, fontFamily: font.family },
  label: { fontSize: 13, fontWeight: '600', color: C.text2, fontFamily: font.family },
  tab: { paddingHorizontal: 14, height: 34, borderRadius: 17, justifyContent: 'center', backgroundColor: 'rgba(255,255,255,0.07)' },
  tabOn: { backgroundColor: C.accentDeep },
  tabText: { fontSize: 13, fontWeight: '700', color: C.text2, fontFamily: font.family },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  tile: { padding: 3, borderRadius: 18, borderWidth: 2, borderColor: 'transparent', alignItems: 'center' },
  tileLabel: { position: 'absolute', bottom: 2, left: 0, right: 0, textAlign: 'center' },
  colors: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  swatch: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  swatchOn: { borderColor: '#fff', transform: [{ scale: 1.12 }] },
});

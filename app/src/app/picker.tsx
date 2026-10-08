// Card picker, shown as a native sheet (Liquid Glass on iOS 26).
//   target=hand    my hand (full round)          target=dHand   my hand now (mid-round)
//   target=opp     the card an opponent played   target=played  cards of finished tricks
//   target=trick   cards before me in the current trick
import { router, useLocalSearchParams } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { useMemo, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { LinearTransition, ZoomIn, ZoomOut } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { ESCAPE, MERMAID, PIRATE, SKULL_KING, SUITS, TIGRESS, cardName, glyphOf, handOrder, isColored, multiplicity } from '@/solver/cards';
import { fullText, query, type Answer } from '@/solver/engine';
import { countOf, sum, useSolver } from '@/solver/store';
import { seatName, POS, rel } from '@/solver-ui/parts';
import { GlassButton, tick } from '@/ui/controls';
import { Glyph } from '@/ui/Glyph';
import { PlayingCard, lookOf } from '@/ui/PlayingCard';
import { C, font, type } from '@/ui/theme';

type Target = 'hand' | 'opp' | 'dHand' | 'played' | 'trick';

export default function Picker() {
  const { target = 'hand' } = useLocalSearchParams<{ target: Target }>();
  const { s, set, full } = useSolver();
  const insets = useSafeAreaInsets();
  const [gridW, setGridW] = useState(0);
  const tile = gridW ? Math.floor((gridW - 6 * 6) / 7) : 40;

  const answer = useMemo(() => {
    if (target !== 'opp') return null;
    const r = query(fullText(full));
    return r.ok ? (r as Answer) : null;
  }, [target, full]);

  const listKey = ({ hand: 'hand', dHand: 'dHand', played: 'played', trick: 'trick', opp: 'play' } as const)[target];
  const list: number[] = s[listKey];
  const used = target === 'hand' ? s.hand : [...s.dHand, ...s.played, ...s.trick];
  const limit =
    target === 'hand' ? s.round
    : target === 'dHand' ? s.round - sum(s.won)
    : target === 'played' ? 4 * sum(s.won)
    : target === 'trick' ? 3
    : Infinity;
  const oppOptions = new Map((answer?.options ?? []).filter((o) => o.type === 'card').map((o) => [o.value as number, o.n ?? 1]));
  const free = (k: number) => (target === 'opp' ? oppOptions.get(k) ?? 0 : multiplicity(k) - countOf(used, k));
  const canAdd = (k: number) => free(k) > 0 && list.length < limit;

  const title =
    target === 'hand' ? 'Deine Hand'
    : target === 'dHand' ? 'Deine Karten jetzt'
    : target === 'played' ? 'Schon gespielt'
    : target === 'trick' ? 'Aktueller Stich'
    : answer ? `${seatName(answer.toAct, answer.me)} ${POS[rel(answer.toAct, answer.me)]} spielt …` : 'Karte';
  const subtitle =
    target === 'played' ? 'Alle Karten aus den fertigen Stichen, Reihenfolge egal'
    : target === 'trick' ? 'Die Karten vor dir, in der Reihenfolge, wie sie gelegt wurden'
    : target === 'opp' ? 'Ausgegraut: Karten, die er nicht haben kann'
    : 'Tippe die Karten an';

  const pick = (k: number) => {
    if (!canAdd(k)) return;
    tick();
    if (target === 'opp') {
      set({ play: [...s.play, k] });
      router.back();
      return;
    }
    const patch: Record<string, unknown> = { [listKey]: [...list, k] };
    if (target === 'hand') patch.bids = [-1, -1, -1, -1];
    set(patch);
    if (target !== 'played' && list.length + 1 >= limit) setTimeout(() => router.back(), 180);
  };
  const remove = (i: number) => {
    tick();
    const next = list.filter((_, j) => j !== i);
    const patch: Record<string, unknown> = { [listKey]: next };
    if (target === 'hand') patch.bids = [-1, -1, -1, -1];
    if (target === 'trick' && !next.includes(TIGRESS)) patch.tigress = -1;
    set(patch);
  };

  const shownList = target === 'trick' ? list : [...list].sort(handOrder);
  return (
    <View style={[styles.sheet, Platform.OS !== 'ios' && styles.sheetFallback]}>
      <View style={styles.head}>
        <View style={{ flex: 1 }}>
          <Text style={type.title}>{title}</Text>
          <Text style={[type.footnote, { marginTop: 2 }]}>{subtitle}</Text>
        </View>
        {Number.isFinite(limit) && (
          <Text style={[styles.count, list.length === limit && { color: C.good }]}>
            {list.length}/{limit}
          </Text>
        )}
        <GlassButton label="Fertig" prominent onPress={() => router.back()} />
      </View>

      {target !== 'opp' && (
        <Animated.View layout={LinearTransition} style={styles.selection}>
          {shownList.length === 0 && <Text style={type.footnote}>Noch keine Karten</Text>}
          {shownList.map((k, i) => (
            <Animated.View key={`${k}-${i}-${shownList.length}`} entering={ZoomIn.springify()} exiting={ZoomOut} layout={LinearTransition}>
              <Pressable accessibilityLabel={`${cardName(k)} entfernen`} onPress={() => remove(target === 'trick' ? i : list.lastIndexOf(k))}>
                <PlayingCard kind={k} size="xs" />
              </Pressable>
            </Animated.View>
          ))}
        </Animated.View>
      )}

      <ScrollView contentContainerStyle={{ paddingBottom: insets.bottom + 24, gap: 14 }} showsVerticalScrollIndicator={false} onLayout={(e) => setGridW(e.nativeEvent.layout.width)}>
        {SUITS.map((su, si) => (
          <View key={si} style={{ gap: 6 }}>
            <View style={styles.suitLabel}>
              <Glyph name={su.glyph} size={14} color={si === 3 ? C.text2 : lookOf(si * 14).from} />
              <Text style={[styles.suitName, { color: si === 3 ? C.text2 : lookOf(si * 14).from }]}>{su.name}</Text>
              {si === 3 && <Text style={styles.trump}>Trumpf</Text>}
            </View>
            <View style={styles.grid}>
              {Array.from({ length: 14 }, (_, v) => si * 14 + v).map((k) => (
                <Tile key={k} kind={k} size={tile} free={free(k)} disabled={!canAdd(k)} selected={countOf(list, k)} onPress={() => pick(k)} />
              ))}
            </View>
          </View>
        ))}
        <View style={styles.specials}>
          {[SKULL_KING, PIRATE, TIGRESS, MERMAID, ESCAPE].map((k) => (
            <Tile key={k} kind={k} wide free={free(k)} disabled={!canAdd(k)} selected={countOf(list, k)} onPress={() => pick(k)} />
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

function Tile({ kind, free, disabled, selected, wide, size, onPress }: { kind: number; free: number; disabled: boolean; selected: number; wide?: boolean; size?: number; onPress: () => void }) {
  const look = lookOf(kind);
  return (
    <Pressable
      accessibilityLabel={cardName(kind)}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [wide ? styles.tileWide : [styles.tile, { width: size, height: size }], { opacity: disabled && !selected ? 0.18 : 1, transform: [{ scale: pressed ? 0.92 : 1 }] }]}>
      <LinearGradient colors={[look.from, look.to]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={[StyleSheet.absoluteFill, { borderRadius: 12 }]} />
      <LinearGradient colors={['rgba(255,255,255,0.3)', 'rgba(255,255,255,0)']} locations={[0, 0.55]} style={[StyleSheet.absoluteFill, { borderRadius: 12 }]} />
      {isColored(kind) ? (
        <Text style={[styles.tileNum, { color: look.ink }]}>{(kind % 14) + 1}</Text>
      ) : (
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
          <Glyph name={glyphOf(kind)} size={20} color={look.ink} />
          <Text style={[styles.tileName, { color: look.ink }]}>{cardName(kind)}</Text>
        </View>
      )}
      {selected > 0 && (
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{selected > 1 ? `×${selected}` : '✓'}</Text>
        </View>
      )}
      {wide && free > 1 && !selected && <Text style={[styles.freeText, { color: look.ink }]}>×{free}</Text>}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  sheet: { flex: 1, paddingHorizontal: 18, paddingTop: 22 },
  sheetFallback: { backgroundColor: 'rgba(20,22,30,0.96)' },
  head: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 14 },
  count: { fontSize: 15, fontWeight: '700', color: C.text2, fontFamily: font.family, ...font.tabular },
  selection: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, minHeight: 46, alignItems: 'center', marginBottom: 14 },
  suitLabel: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  suitName: { fontSize: 14, fontWeight: '700', fontFamily: font.family },
  trump: { fontSize: 11, fontWeight: '700', color: C.text3, marginLeft: 4, fontFamily: font.family },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  tile: { borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  tileWide: { flexGrow: 1, minWidth: '46%', height: 50, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  tileNum: { fontSize: 18, fontWeight: '800', fontFamily: font.family, ...font.tabular },
  tileName: { fontSize: 14, fontWeight: '700', fontFamily: font.family },
  specials: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  badge: { position: 'absolute', top: -5, right: -5, minWidth: 20, height: 20, paddingHorizontal: 4, borderRadius: 10, backgroundColor: C.accentDeep, alignItems: 'center', justifyContent: 'center', borderWidth: 1.5, borderColor: '#0b0d14' },
  badgeText: { fontSize: 11, fontWeight: '800', color: '#fff', fontFamily: font.family },
  freeText: { position: 'absolute', right: 10, fontSize: 12, fontWeight: '700', opacity: 0.7, fontFamily: font.family },
});

// Building blocks of the solver screen: bid advice, seat helpers, hand row,
// move advice headline, table.
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, FadeInDown, LinearTransition, useAnimatedStyle, withSpring, ZoomIn } from 'react-native-reanimated';
import { SUITS, cardName, handOrder } from '@/solver/cards';
import type { Answer, Option, Trick } from '@/solver/engine';
import { tick } from '@/ui/controls';
import { Glass } from '@/ui/Glass';
import { Glyph } from '@/ui/Glyph';
import { PlayingCard, pctText } from '@/ui/PlayingCard';
import { C, font, type } from '@/ui/theme';

export const POS = ['', 'links', 'gegenüber', 'rechts'];
export const seatName = (seat: number, me: number) => (seat === me ? 'Du' : `Sitz ${seat + 1}`);
export const rel = (seat: number, me: number) => (seat - me + 4) % 4;

export function BidAdvice({ advice, chosen, onChoose }: { advice: { bid: number; p: number }[]; chosen?: number; onChoose?: (b: number) => void }) {
  const best = advice.reduce((a, b) => (b.p > a.p ? b : a), advice[0]);
  const mixed = advice.filter((a) => a.p >= 0.05).length > 1;
  return (
    <View>
      <View style={styles.adviceHead}>
        <View>
          <Text style={styles.kicker}>GTO-ANSAGE</Text>
          <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: 10 }}>
            <Text style={styles.bigNum}>{best.bid}</Text>
            <Text style={type.callout}>
              {best.bid === 1 ? 'Stich' : 'Stiche'} · <Text style={{ color: C.text, fontWeight: '700' }}>{pctText(best.p)}%</Text>
            </Text>
          </View>
        </View>
        {mixed && (
          <Glass radius={999} style={styles.mixedPill}>
            <Text style={styles.mixedText}>gemischt</Text>
          </Glass>
        )}
      </View>
      <View style={{ gap: 6, marginTop: 10 }}>
        {advice.map((a, i) => (
          <Bar key={a.bid} label={String(a.bid)} p={a.p} best={a.bid === best.bid} chosen={a.bid === chosen} delay={i * 30} onPress={onChoose && (() => onChoose(a.bid))} />
        ))}
      </View>
      {mixed && (
        <Text style={[type.footnote, { marginTop: 12 }]}>
          GTO mischt hier mehrere Ansagen. Jede ist gleich gut, wenn du sie im richtigen Verhältnis spielst. So bist du nicht ausrechenbar.
        </Text>
      )}
    </View>
  );
}

function Bar({ label, p, best, chosen, delay, onPress }: { label: string; p: number; best: boolean; chosen: boolean; delay: number; onPress?: () => void }) {
  const fill = useAnimatedStyle(() => ({ transform: [{ scaleX: withSpring(Math.max(0.006, p), { damping: 20, stiffness: 140 }) }] }), [p]);
  return (
    <Animated.View entering={FadeIn.delay(delay)}>
      <Pressable
        disabled={!onPress}
        onPress={() => {
          tick();
          onPress?.();
        }}
        accessibilityLabel={`Ansage ${label}: ${pctText(p)} Prozent`}
        style={({ pressed }) => [styles.barRow, chosen && styles.barChosen, pressed && { opacity: 0.7 }]}>
        <Text style={[styles.barLabel, best && { color: C.accent }]}>{label}</Text>
        <View style={styles.barTrack}>
          <Animated.View style={[styles.barFill, best && styles.barFillBest, fill]} />
        </View>
        <Text style={styles.barPct}>{pctText(p)}%</Text>
      </Pressable>
    </Animated.View>
  );
}

/** "GTO spielt Gelb 14 · 72%" */
export function MoveAdvice({ options }: { options: Option[] }) {
  const best = options.reduce((x, y) => ((y.p ?? 0) > (x.p ?? 0) ? y : x), options[0]);
  if (!best || best.p === undefined) return null;
  const name = best.type === 'tigress' ? (best.value === 'pirate' ? 'Tigress als Pirat' : 'Tigress als Flucht') : cardName(best.value as number);
  return (
    <Animated.View entering={FadeInDown.springify()} style={styles.move}>
      <Text style={styles.kicker}>DU BIST DRAN</Text>
      <Text style={styles.moveText}>
        GTO spielt <Text style={{ color: C.accent }}>{name}</Text>
        <Text style={styles.movePct}>  {pctText(best.p)}%</Text>
      </Text>
    </Animated.View>
  );
}

/** My hand with probabilities when I am to act. */
export function HandRow({ hand, options, onPlay, size = 'lg' }: { hand: number[]; options?: Option[]; onPlay?: (k: number) => void; size?: 'lg' | 'md' }) {
  const byKind = new Map((options ?? []).filter((o) => o.type === 'card').map((o) => [o.value as number, o] as const));
  const best = options?.length ? options.reduce((x, y) => ((y.p ?? 0) > (x.p ?? 0) ? y : x), options[0]) : undefined;
  const sorted = [...hand].sort(handOrder);
  return (
    <Animated.View layout={LinearTransition.springify()} style={styles.hand}>
      {sorted.map((k, i) => {
        const o = byKind.get(k);
        return (
          <Animated.View key={`${k}-${i}`} entering={ZoomIn.delay(i * 25).springify()} layout={LinearTransition.springify()}>
            <PlayingCard
              kind={k}
              size={size}
              p={options ? o?.p ?? (o ? 0 : undefined) : undefined}
              best={!!o && o === best}
              dim={!!options && options.length > 0 && !o}
              onPress={o && onPlay ? () => onPlay(k) : undefined}
            />
          </Animated.View>
        );
      })}
    </Animated.View>
  );
}

const PLACE = ['bottom', 'left', 'top', 'right'] as const;

export function Table({ a }: { a: Answer }) {
  const tricks = a.tricks ?? [];
  const last: Trick | undefined = tricks[tricks.length - 1];
  const shown = last && (last.winner < 0 || last.cards.length === 4) ? last : undefined;
  const won = a.won ?? [0, 0, 0, 0];
  const bids = a.bids ?? [0, 0, 0, 0];
  return (
    <View style={styles.table}>
      <View style={styles.felt} />
      {[0, 1, 2, 3].map((s) => {
        const r = rel(s, a.me);
        const turn = a.phase === 'playing' && a.toAct === s;
        const hit = won[s] === bids[s];
        return (
          <Glass key={s} radius={18} tint={turn ? 'rgba(10,132,255,0.55)' : undefined} style={[styles.seat, seatPos[PLACE[r]]]}>
            <Text style={styles.seatName} numberOfLines={1}>
              {seatName(s, a.me)}
              {r > 0 && <Text style={styles.seatPos}> {POS[r]}</Text>}
            </Text>
            <Text style={[styles.seatScore, hit && { color: C.good }, won[s] > bids[s] && { color: C.bad }]}>
              {won[s]}
              <Text style={{ color: C.text3 }}>/{bids[s]}</Text>
            </Text>
            {!!a.voids?.[s] && (
              <View style={styles.voids}>
                {SUITS.map((su, i) =>
                  (a.voids![s] >> i) & 1 ? (
                    <View key={i} style={styles.void}>
                      <Glyph name={su.glyph} size={11} color={C.text2} />
                      <View style={styles.voidSlash} />
                    </View>
                  ) : null,
                )}
              </View>
            )}
          </Glass>
        );
      })}
      <View style={styles.trick} pointerEvents="none">
        {shown?.cards.map(([seat, kind]) => {
          const r = rel(seat, a.me);
          return (
            <Animated.View
              key={`${tricks.length}-${seat}`}
              entering={ZoomIn.springify().damping(16)}
              style={[styles.trickCard, trickPos[PLACE[r]], shown.winner === seat && styles.winner]}>
              <PlayingCard kind={kind} size="sm" />
              {kind === 59 && shown.tigress && <Text style={styles.tig}>{shown.tigress === 'pirate' ? 'Pirat' : 'Flucht'}</Text>}
            </Animated.View>
          );
        })}
        {!shown && a.phase === 'playing' && <Text style={styles.trickEmpty}>Stich {tricks.length + 1}</Text>}
      </View>
    </View>
  );
}

const seatPos = StyleSheet.create({
  bottom: { bottom: 0, alignSelf: 'center' },
  top: { top: 0, alignSelf: 'center' },
  left: { left: 0, top: '38%' },
  right: { right: 0, top: '38%' },
});
const trickPos = StyleSheet.create({
  bottom: { bottom: 0, left: 42 },
  top: { top: 0, left: 42 },
  left: { left: 0, top: 44 },
  right: { right: 0, top: 44 },
});

const styles = StyleSheet.create({
  kicker: { fontSize: 12, fontWeight: '700', letterSpacing: 1.2, color: C.accent, fontFamily: font.family },
  adviceHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  bigNum: { fontSize: 64, fontWeight: '800', color: C.text, letterSpacing: -2, fontFamily: font.family, ...font.tabular },
  mixedPill: { paddingHorizontal: 12, paddingVertical: 6 },
  mixedText: { fontSize: 12, fontWeight: '700', color: C.gold, fontFamily: font.family },
  barRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 4, paddingHorizontal: 6, borderRadius: 12 },
  barChosen: { backgroundColor: 'rgba(255,255,255,0.08)' },
  barLabel: { width: 22, textAlign: 'center', fontSize: 16, fontWeight: '700', color: C.text2, fontFamily: font.family, ...font.tabular },
  barTrack: { flex: 1, height: 10, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.08)', overflow: 'hidden' },
  barFill: { height: '100%', width: '100%', borderRadius: 5, backgroundColor: 'rgba(90,200,250,0.35)', transformOrigin: 'left' },
  barFillBest: { backgroundColor: C.accent, shadowColor: C.accent, shadowOpacity: 0.8, shadowRadius: 8 },
  barPct: { width: 50, textAlign: 'right', fontSize: 13, fontWeight: '600', color: C.text2, fontFamily: font.family, ...font.tabular },
  move: { gap: 4 },
  moveText: { fontSize: 22, fontWeight: '700', color: C.text, letterSpacing: -0.3, fontFamily: font.family },
  movePct: { fontSize: 16, fontWeight: '600', color: C.text2, ...font.tabular },
  hand: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, justifyContent: 'center' },
  table: { height: 300, marginHorizontal: -4 },
  felt: {
    position: 'absolute',
    left: 20,
    right: 20,
    top: 34,
    bottom: 34,
    borderRadius: 999,
    backgroundColor: 'rgba(90,200,250,0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  seat: { position: 'absolute', minWidth: 88, paddingHorizontal: 12, paddingVertical: 7, alignItems: 'center', zIndex: 1 },
  seatName: { fontSize: 13, fontWeight: '700', color: C.text, fontFamily: font.family },
  seatPos: { fontWeight: '500', color: C.text3, fontSize: 12 },
  seatScore: { fontSize: 16, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
  voids: { flexDirection: 'row', gap: 4, marginTop: 2 },
  void: { justifyContent: 'center' },
  voidSlash: { position: 'absolute', left: -1, right: -1, height: 1.5, backgroundColor: C.bad, transform: [{ rotate: '-35deg' }] },
  trick: { position: 'absolute', left: '50%', top: '50%', width: 124, height: 146, marginLeft: -62, marginTop: -73, zIndex: 2 },
  trickCard: { position: 'absolute', alignItems: 'center' },
  winner: { shadowColor: C.good, shadowOpacity: 0.9, shadowRadius: 14 },
  tig: { marginTop: 2, fontSize: 10, fontWeight: '800', color: C.gold, fontFamily: font.family },
  trickEmpty: { position: 'absolute', top: 60, left: 0, right: 0, textAlign: 'center', fontSize: 15, fontWeight: '600', color: C.text3, fontFamily: font.family },
});

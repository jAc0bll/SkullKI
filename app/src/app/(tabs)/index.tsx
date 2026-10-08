import { router } from 'expo-router';
import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { TIGRESS, TIGRESS_ESCAPE, TIGRESS_PIRATE, SUITS } from '@/solver/cards';
import { directText, fullText, isReady, loadRound, query, type Answer, type Result } from '@/solver/engine';
import { sum, useSolver } from '@/solver/store';
import { BidAdvice, HandRow, MoveAdvice, POS, Table, rel, seatName } from '@/solver-ui/parts';
import { Backdrop } from '@/ui/Backdrop';
import { ChipRow, GlassButton, Icon, Panel, SectionTitle, Segmented, Stepper } from '@/ui/controls';
import { Glass, GlassGroup } from '@/ui/Glass';
import { Glyph } from '@/ui/Glyph';
import { PlayingCard } from '@/ui/PlayingCard';
import { C, font, type } from '@/ui/theme';

const pickCards = (target: string) => router.push({ pathname: '/picker', params: { target } });

export default function SolverScreen() {
  const { s, set, undo, reset, canUndo, full, direct } = useSolver();
  const insets = useSafeAreaInsets();
  const [ready, setReady] = useState(isReady(s.round));
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setReady(isReady(s.round));
    setLoadError(null);
    loadRound(s.round).then(
      () => live && setReady(true),
      (e: Error) => live && setLoadError(e.message),
    );
    return () => {
      live = false;
    };
  }, [s.round]);

  const fullReady = s.hand.length === s.round;
  const directReady = s.dHand.length > 0;
  const result: Result | null = useMemo(() => {
    if (!ready) return null;
    if (s.mode === 'full') return fullReady ? query(fullText(full)) : null;
    return directReady ? query(directText(direct)) : null;
  }, [ready, s.mode, fullReady, directReady, full, direct]);
  const a = result?.ok ? (result as Answer) : null;

  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView
        contentInsetAdjustmentBehavior="automatic"
        contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 110, paddingHorizontal: 16, gap: 16 }}
        showsVerticalScrollIndicator={false}>
        {/* header with grouped glass actions */}
        <View style={styles.header}>
          <View>
            <Text style={type.largeTitle}>Solver</Text>
            <Text style={type.footnote}>Skull King · GTO · 4 Spieler</Text>
          </View>
          <GlassGroup spacing={8} style={{ flexDirection: 'row', gap: 8 }}>
            <GlassButton accessibilityLabel="Rückgängig" disabled={!canUndo} onPress={undo} icon={<Icon sf="arrow.uturn.backward" web="↶" />} />
            <GlassButton accessibilityLabel="Neu" onPress={() => reset()} icon={<Icon sf="arrow.counterclockwise" web="⟲" />} />
          </GlassGroup>
        </View>

        <Segmented
          value={s.mode}
          onChange={(mode) => set({ mode })}
          options={[
            { value: 'full', label: 'Ganze Runde' },
            { value: 'direct', label: 'Mitten in der Runde' },
          ]}
        />

        {(s.mode === 'direct' || s.play.length === 0) && (
          <Panel>
            <Text style={styles.label}>Runde</Text>
            <ChipRow
              values={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
              value={s.round}
              onChange={(round) =>
                set({ round, hand: s.hand.slice(0, round), bids: [-1, -1, -1, -1], play: [], dHand: [], won: [0, 0, 0, 0], played: [], trick: [], tigress: -1 })
              }
            />
            <Text style={[styles.label, { marginTop: 16 }]}>Dein Platz in dieser Runde</Text>
            <Segmented
              value={s.me}
              onChange={(me) => set({ me, bids: [-1, -1, -1, -1] })}
              options={[0, 1, 2, 3].map((i) => ({ value: i, label: `${i + 1}.`, sub: i === 0 ? 'beginnt' : i === 3 ? 'zuletzt' : undefined }))}
            />
          </Panel>
        )}

        {!ready && !loadError && (fullReady || directReady) && (
          <Panel style={styles.loading}>
            <ActivityIndicator color={C.accent} />
            <Text style={type.callout}>Strategie für Runde {s.round} wird geladen…</Text>
          </Panel>
        )}
        {loadError && <ErrorBox text={loadError} />}
        {result && !result.ok && <ErrorBox text={result.error} onUndo={canUndo ? undo : undefined} />}

        {s.mode === 'full' ? <FullMode a={a} /> : <DirectMode a={a} />}
      </ScrollView>
    </View>
  );
}

function ErrorBox({ text, onUndo }: { text: string; onUndo?: () => void }) {
  return (
    <Animated.View entering={FadeInDown.springify()}>
      <Glass radius={22} tint="rgba(255,69,58,0.35)" style={styles.error}>
        <Text style={[type.callout, { color: C.text, flex: 1 }]}>{text}</Text>
        {onUndo && <GlassButton label="Zurück" onPress={onUndo} />}
      </Glass>
    </Animated.View>
  );
}

/* ---------------- whole round ---------------- */
function FullMode({ a }: { a: Answer | null }) {
  const { s, set, reset } = useSolver();
  const handFull = s.hand.length === s.round;
  const playing = a && a.phase !== 'bidding';
  const myTurn = !!a && a.phase === 'playing' && a.toAct === a.me;
  const play = (x: number) => set({ play: [...s.play, x] });

  return (
    <>
      {s.play.length === 0 && (
        <Panel>
          <SectionTitle right={<Text style={[styles.count, handFull && { color: C.good }]}>{s.hand.length}/{s.round}</Text>}>Deine Hand</SectionTitle>
          {s.hand.length > 0 ? <HandRow hand={s.hand} size="md" /> : <Text style={type.footnote}>Welche Karten hast du bekommen?</Text>}
          <GlassButton style={{ marginTop: 14, alignSelf: 'flex-start' }} label={handFull ? 'Hand ändern' : 'Karten wählen'} icon={<Icon sf="plus.rectangle.on.rectangle" web="＋" size={16} />} onPress={() => pickCards('hand')} />
        </Panel>
      )}

      {a?.bidAdvice && s.play.length === 0 && (
        <Animated.View entering={FadeInDown.springify()}>
          <Panel>
            <BidAdvice advice={a.bidAdvice} chosen={s.bids[s.me] >= 0 ? s.bids[s.me] : undefined} onChoose={(b) => set({ bids: s.bids.map((v, i) => (i === s.me ? b : v)) })} />
            <View style={styles.divider} />
            <Text style={styles.label}>Ansagen am Tisch</Text>
            {[0, 1, 2, 3].map((seat) => (
              <View key={seat} style={[styles.seatRow, seat === s.me && styles.seatRowMe]}>
                <Text style={type.headline}>
                  {seatName(seat, s.me)}
                  {seat !== s.me && <Text style={type.footnote}>  {POS[rel(seat, s.me)]}</Text>}
                </Text>
                <Stepper unknown value={s.bids[seat]} max={s.round} onChange={(v) => set({ bids: s.bids.map((x, i) => (i === seat ? v : x)) })} />
              </View>
            ))}
            {a.phase === 'bidding' && <Text style={[type.footnote, { marginTop: 10 }]}>Trag alle vier Ansagen ein, dann geht es los.</Text>}
          </Panel>
        </Animated.View>
      )}

      {playing && a && (
        <Animated.View entering={FadeInDown.springify()}>
          <Panel style={{ gap: 16 }}>
            <Table a={a} />

            {a.phase === 'done' && a.points && (
              <View style={{ gap: 10 }}>
                <Text style={styles.kicker}>RUNDE VORBEI</Text>
                <View style={styles.points}>
                  {a.points.map((pts, seat) => (
                    <Glass key={seat} radius={18} tint={seat === a.me ? 'rgba(10,132,255,0.45)' : undefined} style={styles.pointBox}>
                      <Text style={type.footnote}>{seatName(seat, a.me)}</Text>
                      <Text style={[styles.pointNum, { color: pts >= 0 ? C.good : C.bad }]}>{pts > 0 ? `+${pts}` : pts}</Text>
                    </Glass>
                  ))}
                </View>
                <GlassButton prominent label="Nächste Runde" onPress={() => reset({ round: Math.min(10, s.round + 1) })} />
              </View>
            )}

            {a.phase === 'playing' && a.pendingTigress && (
              <View style={{ gap: 10 }}>
                <Text style={styles.kicker}>{myTurn ? 'DEINE TIGRESS' : `${seatName(a.toAct, a.me).toUpperCase()} SPIELT TIGRESS`}</Text>
                <GlassGroup spacing={10} style={{ flexDirection: 'row', gap: 10 }}>
                  {a.options.map((o) => (
                    <GlassButton
                      key={String(o.value)}
                      style={{ flex: 1 }}
                      prominent={myTurn && o === a.options.reduce((x, y) => ((y.p ?? 0) > (x.p ?? 0) ? y : x))}
                      label={`${o.value === 'pirate' ? 'Pirat' : 'Flucht'}${o.p !== undefined ? `  ${Math.round(o.p * 100)}%` : ''}`}
                      onPress={() => play(o.value === 'pirate' ? TIGRESS_PIRATE : TIGRESS_ESCAPE)}
                    />
                  ))}
                </GlassGroup>
              </View>
            )}

            {a.phase === 'playing' && !a.pendingTigress && myTurn && <MoveAdvice options={a.options} />}
            {a.phase === 'playing' && !a.pendingTigress && !myTurn && (
              <GlassButton
                prominent
                label={`Karte von ${seatName(a.toAct, a.me)} eintragen`}
                icon={<Icon sf="hand.tap" web="☝" size={16} color="#fff" />}
                onPress={() => pickCards('opp')}
              />
            )}
            {a.phase === 'playing' && (
              <HandRow hand={a.hand ?? []} options={myTurn && !a.pendingTigress ? a.options : undefined} onPlay={play} />
            )}
          </Panel>
        </Animated.View>
      )}
    </>
  );
}

/* ---------------- mid-round entry ---------------- */
function DirectMode({ a }: { a: Answer | null }) {
  const { s, set } = useSolver();
  const done = sum(s.won);
  const handNeed = s.round - done;
  const counter = (n: number, need: number) => <Text style={[styles.count, n === need && { color: C.good }]}>{n}/{need}</Text>;

  return (
    <>
      <Panel>
        <SectionTitle>Ansagen & Stiche</SectionTitle>
        <View style={styles.gridHead}>
          <Text style={[styles.label, { flex: 1 }]} />
          <Text style={[styles.label, styles.colHead]}>Ansage</Text>
          <Text style={[styles.label, styles.colHead]}>Stiche</Text>
        </View>
        {[0, 1, 2, 3].map((seat) => (
          <View key={seat} style={[styles.seatRow, seat === s.me && styles.seatRowMe]}>
            <Text style={[type.headline, { flex: 1 }]} numberOfLines={1}>
              {seatName(seat, s.me)}
            </Text>
            <Stepper value={s.dBids[seat]} max={s.round} onChange={(v) => set({ dBids: s.dBids.map((x, i) => (i === seat ? v : x)) })} />
            <View style={{ width: 8 }} />
            <Stepper value={s.won[seat]} max={s.round - 1} onChange={(v) => set({ won: s.won.map((x, i) => (i === seat ? v : x)) })} />
          </View>
        ))}
      </Panel>

      <Panel>
        <SectionTitle right={counter(s.played.length, 4 * done)}>Schon gespielt</SectionTitle>
        <Text style={[type.footnote, { marginBottom: 10 }]}>Alle Karten aus den {done} fertigen Stichen</Text>
        <CardStrip cards={s.played} />
        <GlassButton style={styles.pickBtn} label="Bearbeiten" disabled={done === 0} onPress={() => pickCards('played')} />
      </Panel>

      <Panel>
        <SectionTitle right={<Text style={styles.count}>{s.trick.length}/3</Text>}>Aktueller Stich</SectionTitle>
        <Text style={[type.footnote, { marginBottom: 10 }]}>
          {s.trick.length === 0 ? 'Du spielst aus' : `Vor dir gelegt, ausgespielt von ${seatName((s.me - s.trick.length + 4) % 4, s.me)}`}
        </Text>
        <CardStrip cards={s.trick} ordered />
        {s.trick.includes(TIGRESS) && (
          <View style={{ marginTop: 12 }}>
            <Segmented
              value={s.tigress}
              onChange={(tigress) => set({ tigress })}
              options={[
                { value: 1, label: 'Tigress = Pirat' },
                { value: 0, label: 'Tigress = Flucht' },
              ]}
            />
          </View>
        )}
        <GlassButton style={styles.pickBtn} label="Bearbeiten" onPress={() => pickCards('trick')} />
      </Panel>

      <Panel>
        <SectionTitle right={counter(s.dHand.length, handNeed)}>Deine Karten</SectionTitle>
        {s.dHand.length > 0 ? <HandRow hand={s.dHand} size="md" /> : <Text style={type.footnote}>Was hast du noch auf der Hand?</Text>}
        <GlassButton style={styles.pickBtn} label="Bearbeiten" onPress={() => pickCards('dHand')} />
      </Panel>

      <Panel>
        <SectionTitle>Farbe nicht bedient</SectionTitle>
        <Text style={[type.footnote, { marginBottom: 10 }]}>Optional: wer hat eine Farbe schon nicht bedient?</Text>
        {[0, 1, 2, 3].filter((x) => x !== s.me).map((seat) => (
          <View key={seat} style={styles.voidRow}>
            <Text style={[type.callout, { flex: 1, color: C.text }]}>{seatName(seat, s.me)}</Text>
            <GlassGroup spacing={6} style={{ flexDirection: 'row', gap: 6 }}>
              {SUITS.map((su, i) => {
                const on = ((s.voids[seat] >> i) & 1) === 1;
                return (
                  <GlassButton
                    key={i}
                    accessibilityLabel={`${seatName(seat, s.me)} hat kein ${su.name}`}
                    prominent={on}
                    onPress={() => set({ voids: s.voids.map((v, j) => (j === seat ? v ^ (1 << i) : v)) })}
                    icon={<Glyph name={su.glyph} size={17} color={on ? '#fff' : C.text2} />}
                  />
                );
              })}
            </GlassGroup>
          </View>
        ))}
      </Panel>

      {a && (
        <Animated.View entering={FadeInDown.springify()}>
          <Panel style={{ gap: 16 }}>
            <MoveAdvice options={a.options} />
            <HandRow hand={a.hand ?? []} options={a.options} />
          </Panel>
        </Animated.View>
      )}
    </>
  );
}

function CardStrip({ cards, ordered }: { cards: number[]; ordered?: boolean }) {
  if (!cards.length) return <Text style={[type.footnote, { color: C.text3 }]}>keine</Text>;
  return (
    <View style={styles.strip}>
      {cards.map((k, i) => (
        <View key={`${k}-${i}`} style={{ alignItems: 'center' }}>
          <PlayingCard kind={k} size="xs" />
          {ordered && <Text style={styles.order}>{i + 1}</Text>}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 4 },
  label: { fontSize: 13, fontWeight: '600', color: C.text2, marginBottom: 10, fontFamily: font.family },
  kicker: { fontSize: 12, fontWeight: '700', letterSpacing: 1.2, color: C.accent, fontFamily: font.family },
  count: { fontSize: 15, fontWeight: '700', color: C.text2, fontFamily: font.family, ...font.tabular },
  loading: { flexDirection: 'row', gap: 12, alignItems: 'center' },
  error: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: 14 },
  divider: { height: StyleSheet.hairlineWidth, backgroundColor: C.hairline, marginVertical: 18 },
  seatRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 6, paddingHorizontal: 10, borderRadius: 16, marginBottom: 4 },
  seatRowMe: { backgroundColor: 'rgba(90,200,250,0.08)' },
  gridHead: { flexDirection: 'row', paddingHorizontal: 10 },
  colHead: { width: 120, textAlign: 'center', marginBottom: 4 },
  points: { flexDirection: 'row', gap: 8 },
  pointBox: { flex: 1, alignItems: 'center', paddingVertical: 12 },
  pointNum: { fontSize: 24, fontWeight: '800', fontFamily: font.family, ...font.tabular },
  pickBtn: { marginTop: 14, alignSelf: 'flex-start' },
  strip: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  order: { fontSize: 11, fontWeight: '700', color: C.text3, marginTop: 3, fontFamily: font.family },
  voidRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 6 },
});

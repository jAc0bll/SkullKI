// The game screen shared by bot mode and multiplayer.
import * as Haptics from 'expo-haptics';
import { useEffect, useState } from 'react';
import { Platform, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeInUp, LinearTransition } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Avatar } from '@/account/Avatar';
import { cardName, handOrder } from '@/solver/cards';
import { Backdrop } from '@/ui/Backdrop';
import { GlassButton, Icon, tick } from '@/ui/controls';
import { Glass, GlassGroup } from '@/ui/Glass';
import { PlayingCard, pctText } from '@/ui/PlayingCard';
import { PokerTable, TrickPile } from '@/ui/PokerTable';
import { C, font, type } from '@/ui/theme';
import type { GameController, GameTrick, GameView, Player } from './types';

const PLACE = ['bottom', 'left', 'top', 'right'] as const;

export function GameTable({ game, onClose, title }: { game: GameController; onClose: () => void; title: string }) {
  const insets = useSafeAreaInsets();
  const { view, players, error } = game;
  const [hints, setHints] = useState<Record<string, number> | null>(null);
  const [hintsOn, setHintsOn] = useState(false);
  // Let the last trick of a round be seen before the results cover the table.
  const ended = view?.phase === 'roundEnd' || view?.phase === 'gameOver';
  const [showEnd, setShowEnd] = useState(false);
  useEffect(() => {
    if (!ended) return setShowEnd(false);
    const t = setTimeout(() => setShowEnd(true), 1600);
    return () => clearTimeout(t);
  }, [ended, view?.round]);
  const myTurn = !!view && view.toAct.includes(view.seat) && view.legal.length > 0;

  // Refresh hints whenever it becomes my turn.
  useEffect(() => {
    let live = true;
    setHints(null);
    if (hintsOn && myTurn && game.hint) game.hint().then((h) => live && setHints(h));
    return () => {
      live = false;
    };
  }, [hintsOn, myTurn, view, game]);

  useEffect(() => {
    if (myTurn && Platform.OS !== 'web') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Soft);
  }, [myTurn, view?.phase]);

  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <View style={[styles.topBar, { paddingTop: insets.top + 6 }]}>
        <GlassButton accessibilityLabel="Spiel verlassen" onPress={onClose} icon={<Icon sf="xmark" web="✕" size={16} />} />
        <View style={{ flex: 1, alignItems: 'center' }}>
          <Text style={type.headline}>{title}</Text>
          {view && <Text style={type.footnote}>Runde {view.round} von 10</Text>}
        </View>
        {game.hint ? (
          <GlassButton
            accessibilityLabel="GTO-Tipps"
            prominent={hintsOn}
            onPress={() => setHintsOn(!hintsOn)}
            icon={<Icon sf={hintsOn ? 'lightbulb.fill' : 'lightbulb'} web="💡" size={16} color={hintsOn ? '#fff' : C.text} />}
          />
        ) : (
          <View style={{ width: 44 }} />
        )}
      </View>

      {view && <Scores view={view} players={players} />}

      {!view && <Text style={[type.callout, { textAlign: 'center', marginTop: 80 }]}>{error ?? 'Karten werden gemischt…'}</Text>}

      {view && (
        <>
          <GameBoard view={view} players={players} />
          {error && <Text style={[type.footnote, { color: C.bad, textAlign: 'center' }]}>{error}</Text>}
          <View style={[styles.bottom, { paddingBottom: insets.bottom + 12 }]}>
            <Actions view={view} players={players} myTurn={myTurn} hints={hintsOn ? hints : null} act={game.act} />
          </View>
          {ended && showEnd && (
            <RoundEnd view={view} players={players} onNext={game.next} waiting={!!game.waitingNext} onClose={onClose} />
          )}
        </>
      )}
    </View>
  );
}

function Scores({ view, players }: { view: GameView; players: Player[] }) {
  return (
    <GlassGroup spacing={6} style={styles.scores}>
      {[0, 1, 2, 3].map((s) => (
        <Glass key={s} radius={14} tint={s === view.seat ? 'rgba(10,132,255,0.45)' : undefined} style={styles.scoreBox}>
          <Avatar avatar={players[s]?.avatar} size={26} />
          <Text style={styles.scoreName} numberOfLines={1}>
            {s === view.seat ? 'Du' : players[s]?.name ?? `Sitz ${s + 1}`}
          </Text>
          <Text style={styles.scoreNum}>{view.scores[s]}</Text>
        </Glass>
      ))}
    </GlassGroup>
  );
}

function GameBoard({ view, players }: { view: GameView; players: Player[] }) {
  // Fixed size: the table must not jump when the hand or the prompt below changes.
  const { width, height } = useWindowDimensions();
  const boardW = Math.min(width - 16, 520);
  const boardH = Math.round(Math.min(420, Math.max(300, height * 0.42)));
  const rel = (s: number) => (s - view.seat + 4) % 4;
  const trick: GameTrick | null = view.trick.cards.length ? view.trick : view.lastTrick;
  const finished = !!trick && trick === view.lastTrick && trick.winner >= 0;
  const name = (s: number) => (s === view.seat ? 'Du' : players[s]?.name ?? `Sitz ${s + 1}`);
  return (
    <PokerTable width={boardW} height={boardH} style={styles.board}>
      {[0, 1, 2, 3].map((s) => {
        const r = rel(s);
        const turn = (view.phase === 'playing' || view.phase === 'bidding') && view.toAct.includes(s);
        const bidKnown = view.bids[s] >= 0;
        const hit = view.won[s] === view.bids[s];
        return (
          <Glass key={s} radius={18} tint={turn ? 'rgba(10,132,255,0.5)' : undefined} style={[styles.seat, seatPos[PLACE[r]], turn && styles.seatTurn]}>
            <Avatar avatar={players[s]?.avatar} size={30} ring={turn ? C.accent : 'rgba(255,255,255,0.25)'} />
            <View style={{ alignItems: 'flex-start', flexShrink: 1 }}>
              <Text style={styles.seatName} numberOfLines={1}>
                {name(s)}
                {players[s]?.online === false && <Text style={{ color: C.bad }}> ●</Text>}
              </Text>
              {view.phase === 'bidding' ? (
                <Text style={styles.seatSub}>{view.bidIn[s] ? (bidKnown ? `Ansage ${view.bids[s]}` : 'angesagt ✓') : '…'}</Text>
              ) : (
                <Text style={[styles.seatScore, hit && { color: C.good }, view.won[s] > view.bids[s] && { color: C.bad }]}>
                  {view.won[s]}
                  <Text style={{ color: C.text3 }}>/{view.bids[s]}</Text>
                </Text>
              )}
              {s === view.start && <Text style={styles.startTag}>beginnt</Text>}
            </View>
          </Glass>
        );
      })}
      {trick && (
        <TrickPile
          key={`${view.round}-${trick.leader}-${trick.cards[0]?.[0]}-${trick.cards[0]?.[1]}`}
          cards={trick.cards.map(([seat, kind]) => ({
            kind,
            rel: rel(seat),
            label: kind === 59 && trick.tigress ? (trick.tigress === 'pirate' ? 'als Pirat' : 'als Flucht') : undefined,
          }))}
          winner={finished ? trick.cards.findIndex(([seat]) => seat === trick.winner) : undefined}
          caption={finished ? (trick.winner === view.seat ? 'Dein Stich' : `Stich für ${name(trick.winner)}`) : undefined}
        />
      )}
    </PokerTable>
  );
}

function Actions({ view, players, myTurn, hints, act }: { view: GameView; players: Player[]; myTurn: boolean; hints: Record<string, number> | null; act: (a: string) => void }) {
  const hand = [...view.hand].sort(handOrder);
  const legalCards = new Map(view.legal.filter((l) => l.type === 'card').map((l) => [l.value, l.a]));
  const bestHint = hints ? Object.entries(hints).reduce((x, y) => (y[1] > x[1] ? y : x), ['', -1])[0] : '';
  const waitingFor = view.toAct.filter((s) => s !== view.seat).map((s) => players[s]?.name ?? `Sitz ${s + 1}`);

  let prompt: string;
  if (view.phase === 'bidding') prompt = myTurn ? 'Wie viele Stiche machst du?' : `Warte auf ${waitingFor.join(', ')}`;
  else if (view.phase === 'playing') prompt = myTurn ? (view.pendingTigress ? 'Tigress als …' : 'Du bist dran') : `${waitingFor[0] ?? ''} ist dran`;
  else prompt = '';

  return (
    <View style={{ gap: 12 }}>
      {prompt !== '' && (
        <Animated.Text key={prompt} entering={FadeInUp.springify()} style={[styles.prompt, myTurn && { color: C.accent }]}>
          {prompt}
        </Animated.Text>
      )}
      {view.phase === 'bidding' && myTurn && (
        <GlassGroup spacing={6} style={styles.bidRow}>
          {view.legal.map((l) => {
            const p = hints?.[l.a];
            return (
              <View key={l.a} style={{ alignItems: 'center', gap: 4 }}>
                <GlassButton
                  label={String(l.value)}
                  prominent={l.a === bestHint}
                  style={{ minWidth: 48 }}
                  onPress={() => act(l.a)}
                />
                {p !== undefined && <Text style={styles.hintPct}>{pctText(p)}%</Text>}
              </View>
            );
          })}
        </GlassGroup>
      )}
      {view.phase === 'playing' && myTurn && view.pendingTigress && (
        <GlassGroup spacing={10} style={{ flexDirection: 'row', gap: 10 }}>
          {view.legal.map((l) => (
            <GlassButton
              key={l.a}
              style={{ flex: 1 }}
              prominent={l.a === bestHint}
              label={`${l.value === 1 ? 'Pirat' : 'Flucht'}${hints?.[l.a] !== undefined ? `  ${pctText(hints[l.a])}%` : ''}`}
              onPress={() => act(l.a)}
            />
          ))}
        </GlassGroup>
      )}
      <Animated.View layout={LinearTransition.springify()} style={styles.hand}>
        {hand.map((k, i) => {
          const a = legalCards.get(k);
          const playable = myTurn && !view.pendingTigress && view.phase === 'playing' && !!a;
          return (
            <Animated.View key={`${view.round}-${k}-${i}`} entering={FadeInDown.delay(i * 40).springify()} layout={LinearTransition.springify()}>
              <PlayingCard
                kind={k}
                size={hand.length > 7 ? 'md' : 'lg'}
                dim={myTurn && view.phase === 'playing' && !view.pendingTigress && !a}
                best={!!a && a === bestHint}
                p={playable && hints && a ? hints[a] ?? 0 : undefined}
                onPress={playable ? () => act(a!) : undefined}
              />
            </Animated.View>
          );
        })}
      </Animated.View>
    </View>
  );
}

function RoundEnd({ view, players, onNext, waiting, onClose }: { view: GameView; players: Player[]; onNext: () => void; waiting: boolean; onClose: () => void }) {
  const insets = useSafeAreaInsets();
  const res = view.results[view.results.length - 1];
  const over = view.phase === 'gameOver';
  const order = [0, 1, 2, 3].sort((a, b) => view.scores[b] - view.scores[a]);
  const mistakes = view.review.filter((r) => r.round === res.round && r.chosen !== r.best && r.pBest - r.p > 0.15);
  const name = (s: number) => (s === view.seat ? 'Du' : players[s]?.name ?? `Sitz ${s + 1}`);
  const label = (what: string, v: number) => (what === 'bid' ? `Ansage ${v}` : what === 'tigress' ? (v ? 'Tigress als Pirat' : 'Tigress als Flucht') : cardName(v));

  return (
    <Animated.View entering={FadeIn} style={[StyleSheet.absoluteFill, styles.overlay]}>
      <ScrollView contentContainerStyle={{ paddingTop: insets.top + 60, paddingBottom: insets.bottom + 30, paddingHorizontal: 16, gap: 14 }}>
        <Animated.View entering={FadeInDown.springify()}>
          <Glass radius={28} style={styles.sheet}>
            <Text style={styles.kicker}>{over ? 'SPIEL VORBEI' : `RUNDE ${res.round} VORBEI`}</Text>
            {over ? (
              <View style={{ gap: 8, marginTop: 8 }}>
                {order.map((s, i) => (
                  <View key={s} style={[styles.rankRow, s === view.seat && styles.rankMe]}>
                    <Text style={styles.rank}>{i + 1}.</Text>
                    <Text style={[type.headline, { flex: 1 }]}>{name(s)}</Text>
                    <Text style={styles.rankScore}>{view.scores[s]}</Text>
                  </View>
                ))}
              </View>
            ) : (
              <View style={{ marginTop: 8 }}>
                <View style={styles.resHead}>
                  <Text style={[styles.resCell, { flex: 1, textAlign: 'left' }]} />
                  <Text style={styles.resCell}>Ansage</Text>
                  <Text style={styles.resCell}>Stiche</Text>
                  <Text style={styles.resCell}>Punkte</Text>
                  <Text style={styles.resCell}>Gesamt</Text>
                </View>
                {[0, 1, 2, 3].map((s) => (
                  <View key={s} style={[styles.resRow, s === view.seat && styles.rankMe]}>
                    <Text style={[type.callout, { flex: 1, color: C.text, fontSize: 14 }]} numberOfLines={1} adjustsFontSizeToFit>
                      {name(s)}
                    </Text>
                    <Text style={styles.resVal}>{res.bids[s]}</Text>
                    <Text style={styles.resVal}>{res.won[s]}</Text>
                    <Text style={[styles.resVal, { color: res.points[s] >= 0 ? C.good : C.bad }]}>{res.points[s] > 0 ? `+${res.points[s]}` : res.points[s]}</Text>
                    <Text style={styles.resVal}>{view.scores[s]}</Text>
                  </View>
                ))}
              </View>
            )}
          </Glass>
        </Animated.View>

        {!over && (
          <Animated.View entering={FadeInDown.delay(120).springify()}>
            <Glass radius={28} style={styles.sheet}>
              <Text style={styles.kicker}>DEINE ZÜGE IM VERGLEICH ZU GTO</Text>
              {mistakes.length === 0 ? (
                <Text style={[type.callout, { marginTop: 8 }]}>Alles im Rahmen der GTO-Strategie. Stark!</Text>
              ) : (
                mistakes.map((m, i) => (
                  <View key={i} style={styles.mistake}>
                    <Text style={type.footnote}>{m.what === 'bid' ? 'Ansage' : `Stich ${m.trick}`}</Text>
                    <Text style={type.callout}>
                      Du: <Text style={{ color: C.text }}>{label(m.what, m.chosen)}</Text> ({pctText(m.p)}%) · GTO:{' '}
                      <Text style={{ color: C.accent }}>{label(m.what, m.best)}</Text> ({pctText(m.pBest)}%)
                    </Text>
                  </View>
                ))
              )}
            </Glass>
          </Animated.View>
        )}

        {over ? (
          <GlassButton prominent label="Fertig" onPress={onClose} />
        ) : (
          <GlassButton
            prominent
            disabled={waiting}
            label={waiting ? 'Warte auf die anderen…' : 'Weiter zu Runde ' + (res.round + 1)}
            onPress={() => {
              tick();
              onNext();
            }}
          />
        )}
      </ScrollView>
    </Animated.View>
  );
}

const seatPos = StyleSheet.create({
  bottom: { bottom: 0, alignSelf: 'center' },
  top: { top: 0, alignSelf: 'center' },
  left: { left: 0, top: '42%' },
  right: { right: 0, top: '42%' },
});

const styles = StyleSheet.create({
  topBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, gap: 12 },
  scores: { flexDirection: 'row', gap: 6, paddingHorizontal: 16, marginTop: 12 },
  scoreBox: { flex: 1, alignItems: 'center', paddingVertical: 6, gap: 2 },
  scoreName: { fontSize: 11, fontWeight: '600', color: C.text2, fontFamily: font.family },
  scoreNum: { fontSize: 17, fontWeight: '800', color: C.text, fontFamily: font.family, ...font.tabular },
  board: { alignSelf: 'center', marginTop: 8 },
  seat: { position: 'absolute', maxWidth: 150, paddingLeft: 6, paddingRight: 12, paddingVertical: 6, flexDirection: 'row', alignItems: 'center', gap: 8, zIndex: 1 },
  seatTurn: { borderWidth: 1.5, borderColor: C.accent },
  seatName: { fontSize: 13, fontWeight: '700', color: C.text, fontFamily: font.family },
  seatSub: { fontSize: 12, fontWeight: '600', color: C.text2, fontFamily: font.family },
  seatScore: { fontSize: 16, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
  startTag: { fontSize: 9.5, fontWeight: '700', color: C.gold, fontFamily: font.family, marginTop: 1 },
  bottom: { flex: 1, justifyContent: 'flex-end', paddingHorizontal: 12, paddingTop: 8 },
  prompt: { textAlign: 'center', fontSize: 17, fontWeight: '700', color: C.text, fontFamily: font.family },
  bidRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center' },
  hintPct: { fontSize: 11, fontWeight: '700', color: C.text2, fontFamily: font.family, ...font.tabular },
  hand: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center' },
  overlay: { backgroundColor: 'rgba(3,4,8,0.55)' },
  sheet: { padding: 18 },
  kicker: { fontSize: 12, fontWeight: '700', letterSpacing: 1.2, color: C.accent, fontFamily: font.family },
  rankRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, paddingHorizontal: 10, borderRadius: 14 },
  rankMe: { backgroundColor: 'rgba(90,200,250,0.10)' },
  rank: { width: 26, fontSize: 18, fontWeight: '800', color: C.gold, fontFamily: font.family },
  rankScore: { fontSize: 20, fontWeight: '800', color: C.text, fontFamily: font.family, ...font.tabular },
  resHead: { flexDirection: 'row', paddingHorizontal: 8, marginBottom: 4 },
  resRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 7, paddingHorizontal: 8, borderRadius: 12 },
  resCell: { width: 50, textAlign: 'center', fontSize: 11, fontWeight: '600', color: C.text3, fontFamily: font.family },
  resVal: { width: 50, textAlign: 'center', fontSize: 15, fontWeight: '700', color: C.text, fontFamily: font.family, ...font.tabular },
  mistake: { marginTop: 10, gap: 2 },
});

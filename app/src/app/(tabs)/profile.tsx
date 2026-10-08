import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Avatar } from '@/account/Avatar';
import { AvatarGenerator } from '@/account/AvatarGenerator';
import { randomAvatar, type AvatarSpec } from '@/account/avatars';
import { useAccount, type LeaderRow, type User } from '@/account/store';
import { Backdrop } from '@/ui/Backdrop';
import { GlassButton, Panel } from '@/ui/controls';
import { Glass } from '@/ui/Glass';
import { C, font, type } from '@/ui/theme';

export default function ProfileScreen() {
  const insets = useSafeAreaInsets();
  const account = useAccount();
  const { user } = account;
  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView
        contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 110, paddingHorizontal: 16, gap: 16 }}
        keyboardShouldPersistTaps="handled">
        <Text style={type.largeTitle}>Profil</Text>
        {user ? <Signed user={user} /> : <SignIn />}
        <Leaderboard me={user?.name} />
      </ScrollView>
    </View>
  );
}

function SignIn() {
  const { login, error } = useAccount();
  const [name, setName] = useState('');
  const [avatar, setAvatar] = useState<AvatarSpec>(randomAvatar);
  const [busy, setBusy] = useState(false);
  return (
    <Animated.View entering={FadeInDown.springify()} style={{ gap: 16 }}>
      <Panel style={{ gap: 16 }}>
        <Text style={type.title}>Dein Avatar</Text>
        <AvatarGenerator value={avatar} onChange={setAvatar} />
      </Panel>
      <Panel style={{ gap: 12 }}>
        <Text style={type.title}>Dein Name</Text>
        <Text style={type.footnote}>
          Damit meldest du dich an, auch auf anderen Geräten. Ist der Name neu, wird dein Profil angelegt.
        </Text>
        <Glass radius={16} style={styles.inputWrap}>
          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="z. B. Jacob"
            placeholderTextColor={C.text3}
            style={styles.input}
            maxLength={20}
            autoCorrect={false}
            autoCapitalize="words"
          />
        </Glass>
        {error && <Text style={[type.footnote, { color: C.bad }]}>{error}</Text>}
        <GlassButton
          prominent
          label={busy ? 'Einen Moment…' : 'Anmelden'}
          disabled={busy || name.trim().length < 2}
          onPress={async () => {
            setBusy(true);
            await login(name.trim(), avatar);
            setBusy(false);
          }}
        />
      </Panel>
    </Animated.View>
  );
}

function Signed({ user }: { user: User }) {
  const { setAvatar, logout, refresh } = useAccount();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<AvatarSpec>(user.avatar);
  useFocusEffect(
    useCallback(() => {
      refresh();
    }, [refresh]),
  );
  const s = user.stats;
  const pct = (a: number, b: number) => (b ? `${Math.round((100 * a) / b)}%` : '–');
  const tiles: [string, string][] = [
    ['Spiele', String(s.games)],
    ['Siege', String(s.wins)],
    ['Siegquote', pct(s.wins, s.games)],
    ['Ø Punkte', s.games ? String(Math.round(s.points / s.games)) : '–'],
    ['Bestes Spiel', s.best === null ? '–' : String(s.best)],
    ['Ansagen getroffen', pct(s.bidsHit, s.rounds)],
    ['Wie GTO gespielt', pct(s.gtoAgree, s.gtoTotal)],
    ['Gegen KI / Online', `${s.bot.games} / ${s.online.games}`],
  ];
  return (
    <Animated.View entering={FadeInDown.springify()} style={{ gap: 16 }}>
      <Panel style={{ alignItems: 'center', gap: 10 }}>
        <Avatar avatar={user.avatar} size={104} ring="rgba(255,255,255,0.35)" />
        <Text style={type.title}>{user.name}</Text>
        <Text style={type.footnote}>dabei seit {new Date(user.createdAt).toLocaleDateString('de-DE')}</Text>
        <View style={{ flexDirection: 'row', gap: 10, marginTop: 4 }}>
          <GlassButton
            label={editing ? 'Abbrechen' : 'Avatar ändern'}
            onPress={() => {
              setDraft(user.avatar);
              setEditing(!editing);
            }}
          />
          <GlassButton label="Abmelden" onPress={logout} />
        </View>
      </Panel>

      {editing && (
        <Panel style={{ gap: 16 }}>
          <AvatarGenerator value={draft} onChange={setDraft} />
          <GlassButton
            prominent
            label="Speichern"
            onPress={async () => {
              await setAvatar(draft);
              setEditing(false);
            }}
          />
        </Panel>
      )}

      <Panel>
        <Text style={[type.title, { marginBottom: 12 }]}>Statistik</Text>
        <View style={styles.tiles}>
          {tiles.map(([k, v]) => (
            <View key={k} style={styles.tile}>
              <Text style={styles.tileValue}>{v}</Text>
              <Text style={styles.tileLabel}>{k}</Text>
            </View>
          ))}
        </View>
      </Panel>

      <Panel>
        <Text style={[type.title, { marginBottom: 8 }]}>Letzte Spiele</Text>
        {user.history.length === 0 && <Text style={type.footnote}>Noch keine Spiele. Ab ins Spiel!</Text>}
        {user.history.slice(0, 10).map((h, i) => (
          <View key={i} style={styles.histRow}>
            <Text style={[styles.rank, h.rank === 1 && { color: C.gold }]}>{h.rank}.</Text>
            <View style={{ flex: 1 }}>
              <Text style={type.callout} numberOfLines={1}>
                {h.mode === 'bot' ? 'Gegen die KI' : h.players.filter((p) => p !== user.name).join(', ')}
              </Text>
              <Text style={styles.histDate}>{new Date(h.at).toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' })}</Text>
            </View>
            <Text style={[styles.histScore, { color: h.score >= 0 ? C.good : C.bad }]}>{h.score}</Text>
          </View>
        ))}
      </Panel>
    </Animated.View>
  );
}

function Leaderboard({ me }: { me?: string }) {
  const { leaderboard } = useAccount();
  const [rows, setRows] = useState<LeaderRow[] | null>(null);
  useFocusEffect(
    useCallback(() => {
      leaderboard().then(setRows, () => setRows(null));
    }, [leaderboard]),
  );
  if (!rows?.length) return null;
  return (
    <Panel>
      <Text style={[type.title, { marginBottom: 8 }]}>Bestenliste</Text>
      {rows.slice(0, 10).map((r, i) => (
        <View key={r.name} style={[styles.histRow, r.name === me && styles.meRow]}>
          <Text style={[styles.rank, i === 0 && { color: C.gold }]}>{i + 1}.</Text>
          <Avatar avatar={r.avatar} size={32} />
          <Text style={[type.callout, { flex: 1, color: C.text }]} numberOfLines={1}>
            {r.name}
          </Text>
          <Text style={styles.lbStat}>{r.wins} Siege</Text>
          <Text style={styles.lbStat}>Ø {r.avg}</Text>
        </View>
      ))}
    </Panel>
  );
}

const styles = StyleSheet.create({
  inputWrap: { paddingHorizontal: 14, height: 48, justifyContent: 'center' },
  input: { fontSize: 17, color: C.text, fontFamily: font.family, outlineStyle: 'none' } as object,
  tiles: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  tile: { width: '48.5%', padding: 12, borderRadius: 18, backgroundColor: 'rgba(255,255,255,0.05)' },
  tileValue: { fontSize: 24, fontWeight: '800', color: C.text, fontFamily: font.family, fontVariant: ['tabular-nums'] },
  tileLabel: { fontSize: 12, fontWeight: '600', color: C.text2, fontFamily: font.family, marginTop: 2 },
  histRow: { flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 8, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: C.hairline },
  meRow: { backgroundColor: 'rgba(90,200,250,0.08)', borderRadius: 12, paddingHorizontal: 6 },
  rank: { width: 24, fontSize: 16, fontWeight: '800', color: C.text2, fontFamily: font.family },
  histDate: { fontSize: 11, color: C.text3, fontFamily: font.family },
  histScore: { fontSize: 18, fontWeight: '800', fontFamily: font.family, fontVariant: ['tabular-nums'] },
  lbStat: { fontSize: 12, fontWeight: '600', color: C.text2, fontFamily: font.family, minWidth: 54, textAlign: 'right' },
});

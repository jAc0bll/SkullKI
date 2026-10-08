import { router } from 'expo-router';
import { useEffect, useState } from 'react';
import { ScrollView, Share, StyleSheet, Text, TextInput, View } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { GameTable } from '@/game/GameTable';
import { DEFAULT_SERVER, loadPrefs, useOnlineGame } from '@/game/useOnlineGame';
import { Backdrop } from '@/ui/Backdrop';
import { GlassButton, Icon, Panel } from '@/ui/controls';
import { Glass } from '@/ui/Glass';
import { C, font, type } from '@/ui/theme';

export default function Online() {
  const insets = useSafeAreaInsets();
  const net = useOnlineGame();
  const { resume } = net;
  const [name, setName] = useState('');
  const [server, setServer] = useState(DEFAULT_SERVER);
  const [code, setCode] = useState('');
  const [showServer, setShowServer] = useState(false);

  useEffect(() => {
    loadPrefs().then((p) => {
      setName(p.name);
      setServer(p.server);
      resume(p.server);
    });
  }, [resume]);

  const room = net.room;
  if (room?.started) {
    return (
      <GameTable
        game={net.controller}
        title={`Raum ${room.code}`}
        onClose={async () => {
          await net.leave();
          router.back();
        }}
      />
    );
  }

  const canGo = name.trim().length > 0 && net.status !== 'connecting';
  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView
        contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 40, paddingHorizontal: 16, gap: 16 }}
        keyboardShouldPersistTaps="handled">
        <View style={styles.header}>
          <GlassButton accessibilityLabel="Zurück" onPress={() => router.back()} icon={<Icon sf="chevron.left" web="‹" size={16} />} />
          <Text style={type.largeTitle}>Mit Freunden</Text>
        </View>

        {!room && (
          <>
            <Panel style={{ gap: 12 }}>
              <Text style={styles.label}>Dein Name</Text>
              <Glass radius={16} style={styles.inputWrap}>
                <TextInput
                  value={name}
                  onChangeText={setName}
                  placeholder="z. B. Jacob"
                  placeholderTextColor={C.text3}
                  style={styles.input}
                  maxLength={20}
                  autoCorrect={false}
                />
              </Glass>
              <GlassButton prominent label="Raum erstellen" disabled={!canGo} onPress={() => net.create(server, name.trim())} />
            </Panel>
            <Panel style={{ gap: 12 }}>
              <Text style={styles.label}>Raum-Code von einem Freund</Text>
              <View style={{ flexDirection: 'row', gap: 10 }}>
                <Glass radius={16} style={[styles.inputWrap, { flex: 1 }]}>
                  <TextInput
                    value={code}
                    onChangeText={(t) => setCode(t.toUpperCase())}
                    placeholder="ABCD"
                    placeholderTextColor={C.text3}
                    style={[styles.input, styles.codeInput]}
                    maxLength={4}
                    autoCapitalize="characters"
                    autoCorrect={false}
                  />
                </Glass>
                <GlassButton label="Beitreten" disabled={!canGo || code.length !== 4} onPress={() => net.join(server, name.trim(), code)} />
              </View>
            </Panel>
            <Panel style={{ gap: 10 }}>
              <Text style={type.footnote} onPress={() => setShowServer(!showServer)}>
                Server: {server} {showServer ? '▴' : '▾'}
              </Text>
              {showServer && (
                <Glass radius={16} style={styles.inputWrap}>
                  <TextInput value={server} onChangeText={setServer} autoCapitalize="none" autoCorrect={false} style={styles.input} />
                </Glass>
              )}
            </Panel>
          </>
        )}

        {net.error && (
          <Glass radius={18} tint="rgba(255,69,58,0.35)" style={{ padding: 14 }}>
            <Text style={[type.callout, { color: C.text }]}>{net.error}</Text>
          </Glass>
        )}
        {net.status === 'connecting' && <Text style={type.footnote}>Verbinde…</Text>}

        {room && !room.started && (
          <Animated.View entering={FadeInDown.springify()} style={{ gap: 16 }}>
            <Panel style={{ alignItems: 'center', gap: 6 }}>
              <Text style={styles.label}>Raum-Code</Text>
              <Text style={styles.bigCode}>{room.code}</Text>
              <GlassButton
                label="Code teilen"
                icon={<Icon sf="square.and.arrow.up" web="⇪" size={16} />}
                onPress={() => Share.share({ message: `Spiel Skull King mit mir in SkullKI! Raum-Code: ${room.code}` })}
              />
            </Panel>
            <Panel style={{ gap: 8 }}>
              <Text style={styles.label}>Am Tisch</Text>
              {[0, 1, 2, 3].map((s) => {
                const p = room.players[s];
                return (
                  <View key={s} style={[styles.seatRow, s === room.seat && styles.me]}>
                    <Text style={[type.headline, { flex: 1, color: p ? C.text : C.text3 }]}>
                      {p ? p.name + (s === room.seat ? ' (du)' : '') : 'frei, spielt die KI'}
                    </Text>
                    {s === 0 && p && <Text style={styles.host}>Host</Text>}
                  </View>
                );
              })}
            </Panel>
            {room.host ? (
              <GlassButton prominent label="Spiel starten" onPress={net.start} />
            ) : (
              <Text style={[type.callout, { textAlign: 'center' }]}>Warte, bis der Host startet…</Text>
            )}
            <GlassButton label="Raum verlassen" onPress={net.leave} />
          </Animated.View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  label: { fontSize: 13, fontWeight: '600', color: C.text2, fontFamily: font.family },
  inputWrap: { paddingHorizontal: 14, height: 48, justifyContent: 'center' },
  input: { fontSize: 17, color: C.text, fontFamily: font.family, outlineStyle: 'none' } as object,
  codeInput: { letterSpacing: 6, fontWeight: '700' },
  bigCode: { fontSize: 52, fontWeight: '800', letterSpacing: 10, color: C.text, fontFamily: font.family },
  seatRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 14,
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  me: { backgroundColor: 'rgba(90,200,250,0.10)' },
  host: { fontSize: 12, fontWeight: '700', color: C.gold, fontFamily: font.family },
});

import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Backdrop } from '@/ui/Backdrop';
import { STYLES } from '@/account/avatars';
import { Panel } from '@/ui/controls';
import { C, font, type } from '@/ui/theme';

const ROWS: [string, string][] = [
  ['1', '0,00'],
  ['2', '0,14 / 0,19'],
  ['3', '0,19 / 0,05'],
  ['4 – 10', '0,00'],
];

export default function InfoScreen() {
  const insets = useSafeAreaInsets();
  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 110, paddingHorizontal: 16, gap: 16 }}>
        <Text style={type.largeTitle}>Info</Text>
        <Panel style={{ gap: 10 }}>
          <Text style={type.title}>Was ist GTO?</Text>
          <Text style={type.callout}>
            Eine Strategie, die man nicht ausnutzen kann: Egal wie die anderen spielen, sie holen gegen dich auf Dauer nichts heraus. Wie bei Poker-Solvern ist sie oft gemischt: in derselben Lage manchmal A, manchmal B.
          </Text>
        </Panel>
        <Panel style={{ gap: 10 }}>
          <Text style={type.title}>Wie gut ist die KI?</Text>
          <Text style={type.callout}>
            Pro Runde wurde eine Gegenstrategie trainiert, die gezielt Schwächen sucht. So viele Punkte pro Runde holt sie heraus (Sitz 1 / Sitz 4):
          </Text>
          <View style={{ marginTop: 6 }}>
            {ROWS.map(([r, v]) => (
              <View key={r} style={styles.row}>
                <Text style={[type.body, { color: C.text2 }]}>Runde {r}</Text>
                <Text style={styles.value}>{v}</Text>
              </View>
            ))}
          </View>
          <Text style={type.footnote}>0 heißt: keine Schwäche gefunden. Punkte pro Runde liegen typisch bei ±20 bis ±100.</Text>
        </Panel>
        <Panel style={{ gap: 10 }}>
          <Text style={type.title}>Grenzen</Text>
          <Text style={type.callout}>
            Die KI optimiert die Punkte jeder Runde. Dass man bei großem Rückstand in Runde 10 mehr riskieren sollte, weiß sie noch nicht. Gerechnet wird komplett auf deinem iPhone, ohne Internet.
          </Text>
        </Panel>
        <Panel style={{ gap: 10 }}>
          <Text style={type.title}>Deine Daten</Text>
          <Text style={type.callout}>
            Profil: Name, Avatar und Statistik liegen auf dem SkullKI-Server. Gespielte Partien werden zusätzlich anonym
            gespeichert (zufällige Nummer statt Name), um zu lernen, wie Menschen spielen. Das lässt sich im Profil
            abschalten. Der Solver rechnet nur auf deinem Gerät.
          </Text>
        </Panel>
        <Panel style={{ gap: 6 }}>
          <Text style={type.title}>Avatare</Text>
          <Text style={type.footnote}>Erzeugt mit DiceBear (github.com/dicebear/dicebear, MIT). Stile:</Text>
          {STYLES.map((s) => (
            <Text key={s.id} style={type.footnote}>
              · {s.credit}
            </Text>
          ))}
        </Panel>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: C.hairline },
  value: { fontSize: 16, fontWeight: '700', color: C.text, fontFamily: font.family, fontVariant: ['tabular-nums'] },
});

import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Backdrop } from '@/ui/Backdrop';
import { Panel } from '@/ui/controls';
import { Glass } from '@/ui/Glass';
import { PlayingCard } from '@/ui/PlayingCard';
import { C, font, type } from '@/ui/theme';

// Bot mode comes next (docs/APP_PLAN.md); this tab announces it.
export default function PlayScreen() {
  const insets = useSafeAreaInsets();
  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 110, paddingHorizontal: 16, gap: 16 }}>
        <Text style={type.largeTitle}>Spielen</Text>
        <View style={styles.fan}>
          {[60, 58, 59, 57, 45].map((k, i) => (
            <View key={k} style={{ transform: [{ rotate: `${(i - 2) * 9}deg` }, { translateY: Math.abs(i - 2) * 8 }], marginHorizontal: -10 }}>
              <PlayingCard kind={k} size="lg" />
            </View>
          ))}
        </View>
        <Panel style={{ gap: 10 }}>
          <Glass radius={999} tint="rgba(10,132,255,0.5)" style={styles.pill}>
            <Text style={styles.pillText}>Kommt als Nächstes</Text>
          </Glass>
          <Text style={type.title}>Gegen die KI</Text>
          <Text style={type.callout}>
            Du gegen drei Bots, die mit der GTO-Strategie spielen. Komplett offline, alle 10 Runden. Nach jeder Runde siehst du, wo du von GTO abgewichen bist.
          </Text>
        </Panel>
        <Panel style={{ gap: 10 }}>
          <Text style={type.title}>Mit Freunden</Text>
          <Text style={type.callout}>Online-Räume über deinen Server. Freie Plätze übernimmt die KI.</Text>
        </Panel>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  fan: { flexDirection: 'row', justifyContent: 'center', paddingVertical: 24 },
  pill: { alignSelf: 'flex-start', paddingHorizontal: 12, paddingVertical: 6 },
  pillText: { fontSize: 12, fontWeight: '700', color: '#fff', fontFamily: font.family },
});

import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Backdrop } from '@/ui/Backdrop';
import { Icon, tick } from '@/ui/controls';
import { Glass } from '@/ui/Glass';
import { PlayingCard } from '@/ui/PlayingCard';
import { C, font, type } from '@/ui/theme';

export default function PlayScreen() {
  const insets = useSafeAreaInsets();
  return (
    <View style={{ flex: 1 }}>
      <Backdrop />
      <ScrollView contentContainerStyle={{ paddingTop: insets.top + 8, paddingBottom: insets.bottom + 110, paddingHorizontal: 16, gap: 16 }}>
        <Text style={type.largeTitle}>Spielen</Text>
        <View style={styles.fan}>
          {[60, 58, 59, 57, 45].map((k, i) => (
            <Animated.View
              key={k}
              entering={FadeInDown.delay(i * 70).springify()}
              style={{ transform: [{ rotate: `${(i - 2) * 9}deg` }, { translateY: Math.abs(i - 2) * 8 }], marginHorizontal: -10 }}>
              <PlayingCard kind={k} size="lg" />
            </Animated.View>
          ))}
        </View>
        <Choice
          title="Gegen die KI"
          text="Du gegen drei Bots mit der GTO-Strategie. Offline, alle 10 Runden, mit Tipps und Auswertung."
          sf="cpu"
          web="◆"
          onPress={() => router.push('/bot')}
        />
        <Choice
          title="Mit Freunden"
          text="Raum erstellen, Code teilen, zusammen spielen. Freie Plätze übernimmt die KI."
          sf="person.3.fill"
          web="●"
          onPress={() => router.push('/online')}
        />
      </ScrollView>
    </View>
  );
}

function Choice({ title, text, sf, web, onPress }: { title: string; text: string; sf: Parameters<typeof Icon>[0]['sf']; web: string; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={title}
      onPress={() => {
        tick();
        onPress();
      }}
      style={({ pressed }) => ({ transform: [{ scale: pressed ? 0.97 : 1 }] })}>
      <Glass radius={28} interactive style={styles.choice}>
        <View style={styles.choiceIcon}>
          <Icon sf={sf} web={web} size={24} color="#fff" />
        </View>
        <View style={{ flex: 1, gap: 4 }}>
          <Text style={type.title}>{title}</Text>
          <Text style={type.callout}>{text}</Text>
        </View>
        <Icon sf="chevron.right" web="›" size={16} color={C.text3} />
      </Glass>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  fan: { flexDirection: 'row', justifyContent: 'center', paddingVertical: 20 },
  choice: { flexDirection: 'row', alignItems: 'center', gap: 16, padding: 18 },
  choiceIcon: {
    width: 52,
    height: 52,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.accentDeep,

  },
});

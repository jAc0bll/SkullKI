import { TabList, TabSlot, TabTrigger, Tabs, type TabListProps, type TabTriggerSlotProps } from 'expo-router/ui';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Glass } from '@/ui/Glass';
import { C, font } from '@/ui/theme';

// Web: a floating glass tab bar in the style of iOS 26.
export default function TabLayout() {
  return (
    <Tabs>
      <TabSlot style={{ flex: 1 }} />
      <TabList asChild>
        <Bar>
          <TabTrigger name="index" href="/" asChild>
            <Tab label="Solver" glyph="◆" />
          </TabTrigger>
          <TabTrigger name="play" href="/play" asChild>
            <Tab label="Spielen" glyph="●" />
          </TabTrigger>
          <TabTrigger name="profile" href="/profile" asChild>
            <Tab label="Profil" glyph="◉" />
          </TabTrigger>
          <TabTrigger name="info" href="/info" asChild>
            <Tab label="Info" glyph="ⓘ" />
          </TabTrigger>
        </Bar>
      </TabList>
    </Tabs>
  );
}

function Bar(props: TabListProps) {
  return (
    <View style={styles.wrap} pointerEvents="box-none">
      <Glass style={styles.bar}>{props.children}</Glass>
    </View>
  );
}

function Tab({ label, glyph, isFocused, ...props }: TabTriggerSlotProps & { label: string; glyph: string }) {
  return (
    <Pressable {...props} style={[styles.tab, isFocused && styles.tabOn]}>
      <Text style={[styles.glyph, isFocused && { color: C.accent }]}>{glyph}</Text>
      <Text style={[styles.label, isFocused && { color: C.accent }]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: { position: 'absolute', left: 0, right: 0, bottom: 18, alignItems: 'center' },
  bar: { flexDirection: 'row', padding: 4, gap: 2 },
  tab: { alignItems: 'center', paddingHorizontal: 22, paddingVertical: 6, borderRadius: 999 },
  tabOn: { backgroundColor: 'rgba(255,255,255,0.12)' },
  glyph: { fontSize: 16, color: C.text2 },
  label: { fontSize: 11, fontWeight: '600', color: C.text2, fontFamily: font.family },
});

import { NativeTabs } from 'expo-router/unstable-native-tabs';
import { DynamicColorIOS } from 'react-native';

// The system tab bar: real Liquid Glass on iOS 26, floats over the content.
export default function TabLayout() {
  const tint = DynamicColorIOS({ dark: '#5AC8FA', light: '#0A84FF' });
  return (
    <NativeTabs tintColor={tint} minimizeBehavior="onScrollDown">
      <NativeTabs.Trigger name="index">
        <NativeTabs.Trigger.Label>Solver</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf={{ default: 'sparkles.rectangle.stack', selected: 'sparkles.rectangle.stack.fill' }} md="style" />
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="play">
        <NativeTabs.Trigger.Label>Spielen</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf={{ default: 'gamecontroller', selected: 'gamecontroller.fill' }} md="sports_esports" />
      </NativeTabs.Trigger>
      <NativeTabs.Trigger name="info">
        <NativeTabs.Trigger.Label>Info</NativeTabs.Trigger.Label>
        <NativeTabs.Trigger.Icon sf={{ default: 'info.circle', selected: 'info.circle.fill' }} md="info" />
      </NativeTabs.Trigger>
    </NativeTabs>
  );
}

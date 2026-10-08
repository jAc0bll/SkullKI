import { DarkTheme, Stack, ThemeProvider } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { AccountProvider } from '@/account/store';
import { SolverProvider } from '@/solver/store';
import { C } from '@/ui/theme';

SplashScreen.preventAutoHideAsync();

const theme = { ...DarkTheme, colors: { ...DarkTheme.colors, background: C.bg, card: 'transparent', primary: C.accent } };

export default function RootLayout() {
  useEffect(() => {
    SplashScreen.hideAsync();
  }, []);
  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: C.bg }}>
      <ThemeProvider value={theme}>
        <AccountProvider>
        <SolverProvider>
          <StatusBar style="light" />
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="(tabs)" />
            <Stack.Screen name="bot" options={{ presentation: 'fullScreenModal', gestureEnabled: false }} />
            <Stack.Screen name="online" options={{ presentation: 'fullScreenModal', gestureEnabled: false }} />
            {/* native sheet: Liquid Glass on iOS 26 */}
            <Stack.Screen
              name="picker"
              options={{
                presentation: 'formSheet',
                sheetAllowedDetents: [0.62, 1],
                sheetGrabberVisible: true,
                contentStyle: { backgroundColor: 'transparent' },
              }}
            />
          </Stack>
        </SolverProvider>
        </AccountProvider>
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}

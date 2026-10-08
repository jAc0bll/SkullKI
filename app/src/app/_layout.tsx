import { DarkTheme, Stack, ThemeProvider } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
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
        <SolverProvider>
          <StatusBar style="light" />
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="(tabs)" />
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
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}

// Where the SkullKI server lives (accounts API + multiplayer).
// In the browser the app is served by that server itself, so it is the same
// address. For local development: EXPO_PUBLIC_SERVER=http://localhost:8787
import { Platform } from 'react-native';

const sameOriginHttp =
  Platform.OS === 'web' && typeof location !== 'undefined' && !/:8081$/.test(location.host) ? location.origin : null;

export const SERVER_HTTP: string = process.env.EXPO_PUBLIC_SERVER ?? sameOriginHttp ?? 'https://skullki.schwartihost.com';
export const SERVER_WS = SERVER_HTTP.replace(/^http/, 'ws');
/** true when the app comes from the game server (the address is fixed) */
export const FIXED_SERVER = !!sameOriginHttp;

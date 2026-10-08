# SkullKI-App: Plan

Stand 2026-10-08. Eine **native iPhone-App** (React Native / Expo SDK 57) im
Apple-**Liquid-Glass**-Design. Dieselbe App läuft auch im Browser und später auf
Android. Die KI rechnet direkt auf dem Gerät: auf dem iPhone als nativ
kompiliertes C++, im Browser als WebAssembly.

## Aufbau

| Teil | Wo |
|---|---|
| App (Bildschirme, Design) | `app/src` (Expo Router: `app/src/app`) |
| C++-KI im iPhone | `app/modules/sk-solver` (Expo-Modul: Swift → Objective-C++ → `solver/src/spot_c_api.cpp`) |
| C++-KI im Browser | `scripts/build_wasm.sh` → `app/modules/sk-solver/src/web/sk-web.js` |
| Strategie-Netze | `app/assets/models/r1..r10.bin` (float16, 19 MB, `scripts/export_models.py`) |
| iPhone-Build | `.github/workflows/ios.yml` → `SkullKI.ipa` (Installation: `docs/IPHONE.md`) |

Design: Liquid Glass nur für die Bedienebene (Tab-Leiste, Knöpfe, Regler,
Karten-Auswahl als Sheet), wie Apple es vorgibt. Karten und Tisch sind
farbiger Inhalt darunter. Auf iOS 26 ist es Apples echtes Material
(`expo-glass-effect`, native Tab-Leiste und Sheets), sonst eine
Milchglas-Nachbildung.

## Solver-Modi

- **Ganze Runde**: Hand → GTO-Ansage → Ansagen → Karte für Karte mitspielen.
- **Mitten in der Runde**: nur der aktuelle Stand (deine Karten, aktueller
  Stich, Ansagen, Stiche bisher, schon gespielte Karten). Das ergibt exakt
  dieselbe Antwort wie die ganze Runde (getestet), weil die KI nur den
  aktuellen Stand sieht.

## Lokal entwickeln

```bash
npm --prefix app install
npm --prefix app run web
```

C++ geändert? `bash scripts/build_wasm.sh` (Emscripten in `C:\Users\jaxob\emsdk`),
danach `npm --prefix app run check-wasm` (vergleicht mit dem nativen Solver).

## Nächste Schritte

1. ✅ Solver-App (beide Modi), iPhone-Build in der Cloud
2. **Gegen die KI**: offline, 3 Bots spielen nach den GTO-Wahrscheinlichkeiten,
   10 Runden, danach Analyse, wo du von GTO abgewichen bist
3. **Multiplayer** über deinen Linux-Server (nicht P2P: Bei P2P kennt das
   mischende Handy alle Karten). Räume mit Code, freie Plätze übernimmt die KI
4. Feinschliff: Sounds, Spielverlauf speichern, Android-Build
5. Später: 2-Spieler-Modus (eigenes Training), Gewinnwahrscheinlichkeit statt Punkte

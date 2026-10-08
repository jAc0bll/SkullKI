# SkullKI-App: Plan

> **Für normale Nutzer: die Website.** Sideloading ist zu umständlich. Dieselbe App
> läuft im Handy-Browser (https://skullki.schwartihost.com) und lässt sich zum
> Home-Bildschirm hinzufügen. Der Linux-Server liefert Website und Multiplayer
> aus (`server/setup.sh`, Anleitung `docs/MULTIPLAYER_SERVER.md`). Die native
> iPhone-App (`.ipa`) bleibt für dich als Option; für echte Verteilung wäre
> später TestFlight/App Store (Apple-Entwicklerkonto 99 €/Jahr) der Weg.

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

## Spielen

- **Gegen die KI** (`app/src/app/bot.tsx`): offline auf dem iPhone, du + 3 Bots,
  die zufällig nach den GTO-Wahrscheinlichkeiten spielen. Glühbirne = GTO-Tipps,
  nach jeder Runde Auswertung, wo du deutlich von GTO abgewichen bist.
- **Mit Freunden** (`app/src/app/online.tsx` + `server/`): Räume mit 4-Buchstaben-
  Code über deinen Server, freie Plätze spielt die KI, Wiederverbinden nach
  Funkloch. Einrichtung: `docs/MULTIPLAYER_SERVER.md`.
- Beide nutzen dieselbe C++-Spiel-Engine (`solver/src/session.cpp`): in der App
  nativ, auf dem Server als WebAssembly.

## Nächste Schritte

1. ✅ Solver-App, ✅ Bot-Modus, ✅ Multiplayer (Server muss noch auf deinen Linux-Server)
2. Feinschliff: Sounds, Spielverlauf/Statistik speichern, Android-Build
3. Später: 2-Spieler-Modus (eigenes Training), Gewinnwahrscheinlichkeit statt Punkte

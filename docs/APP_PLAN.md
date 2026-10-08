# SkullKI-App: Plan (Spot-Solver → iPhone-App → Bot → Multiplayer)

Stand 2026-10-08. Kurz gesagt: **eine** Web-App für alles. Sie läuft im Browser,
lässt sich auf dem iPhone wie eine echte App installieren (ohne App Store), und
die KI rechnet direkt auf dem Gerät (WebAssembly). Nur Multiplayer braucht einen
Server, und dafür reicht dein Linux-Server.

## 1. Was es schon gibt (Meilenstein 1: Spot-Solver)

- `web/`: React + TypeScript + Vite, PWA (offline-fähig, installierbar)
- `solver/src/spot.cpp`: baut eine Situation aus dem, was *du* weißt (Hand, Platz,
  Ansagen, gespielte Karten), prüft sie auf Regelverstöße und liefert die
  GTO-Wahrscheinlichkeiten. Als WebAssembly kompiliert (166 KB), rechnet
  bit-identisch zum C++-Original (geprüft mit `web/wasm/check.mjs`)
- KI-Netze: `scripts/export_models.py` → `web/public/models/r1..r10.bin`
  (float16, insgesamt 19 MB; geladen wird nur die gerade gebrauchte Runde, 0,6–2,3 MB)

Starten (lokal):

```bash
npm --prefix web install
npm --prefix web run dev
```

Neu bauen nach Änderungen am C++-Code: `bash scripts/build_wasm.sh` (braucht
Emscripten, installiert in `C:\Users\jaxob\emsdk`).

## 2. iPhone-App ohne App Store

| Weg | Aufwand | Haken |
|---|---|---|
| **PWA** (Safari → Teilen → „Zum Home-Bildschirm“) | schon fertig | keiner für unseren Zweck: Vollbild, eigenes Icon, offline, keine Apple-ID nötig |
| Capacitor-Hülle + AltStore/SideStore (kostenlose Apple-ID) | mittel | App muss alle **7 Tage** neu signiert werden, max. 3 Apps |
| Capacitor + Apple-Entwicklerkonto (99 €/Jahr), Sideload/TestFlight | mittel | Kosten, Signieren 1×/Jahr |
| EU-Alternativ-Marktplatz (z. B. AltStore PAL) | hoch | Apple-Notarisierung + Entwicklerkonto trotzdem nötig |

**Entscheidung: PWA.** Gleicher Code wie die Website, Updates kommen automatisch,
kein Ablaufdatum. Voraussetzung: Die Seite muss über **HTTPS** erreichbar sein
(z. B. `skullki.schwartihost.com` über deinen Caddy, der macht HTTPS automatisch).
Capacitor bauen wir nur, falls wir später etwas brauchen, das eine PWA nicht kann.

## 3. Gegen die KI spielen (Meilenstein 2)

- Komplett offline auf dem Gerät: Du + 3 Bots, alle 10 Runden, Punktestand.
- Neue C++-API `game.cpp` (WASM): mischt und verteilt, Bots ziehen ihre Züge
  **zufällig nach den GTO-Wahrscheinlichkeiten** (gemischte Strategie, deshalb
  nicht ausrechenbar).
- Nach jeder Runde: **Analyse**, also wo du von GTO abgewichen bist (Meilenstein 3
  ergänzt, wie viele Punkte das im Schnitt gekostet hat; dafür Rollouts mit dem
  Netz im Hintergrund-Thread).
- Ehrliche Einschränkung: Die Strategie optimiert Punkte *pro Runde*. Dass man
  z. B. in Runde 10 bei großem Rückstand mehr riskieren sollte, kennt sie noch
  nicht → späterer Schritt „Gewinnwahrscheinlichkeit statt Punkte“.

## 4. Multiplayer (Meilenstein 4)

**Empfehlung: über deinen Linux-Server, nicht P2P.**

- Skull King hat verdeckte Karten. Bei P2P muss *ein* Handy mischen und kennt dann
  alle Karten, d. h. man könnte schummeln. Der Server mischt und schickt jedem nur
  seine eigenen Karten.
- P2P (WebRTC) braucht trotzdem einen Server zum Verbinden und scheitert oft an
  Routern/Mobilfunk-NAT.
- Umsetzung: kleiner Node.js-WebSocket-Server (gleiche Technik wie dein
  Challenge-Projekt), hinter Caddy. Räume mit 4-stelligem Code, 2–4 Menschen,
  **freie Plätze füllen Bots** (der Server nutzt dieselbe WebAssembly-KI, die läuft
  auch in Node). Wiederverbinden nach Funkloch.
- Kosten: keine extra, der Server läuft eh.

## 5. Reihenfolge

1. ✅ Spot-Solver (lokal fertig)
2. Online stellen: `web/dist` auf deinen Server (Caddy) oder GitHub Pages → auf dem iPhone installieren
3. Bot-Modus (offline)
4. Analyse mit Punkteverlust pro Entscheidung
5. Multiplayer-Server
6. Feinschliff: Sounds, Haptik, Spielverlauf speichern
7. Später: 2-Spieler-Modus (eigenes Training, echtes GTO), Gewinnwahrscheinlichkeit

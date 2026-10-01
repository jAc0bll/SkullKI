# Training auf einem gemieteten Server

Ziel: alle 10 Runden mit neuronalem CFR trainieren und für jede Runde messen,
wie viel ein trainierter Ausnutzer gegen die Strategie herausholt.

## 1. Server mieten

Empfehlung: **1× NVIDIA-GPU** (RTX 4090 / A5000 / L40S o. ä.), **32+ CPU-Kerne**,
**64 GB RAM**, **150 GB Festplatte**, Ubuntu 22.04 oder 24.04 (z. B. RunPod oder
vast.ai). Mehrere GPUs bringen nichts. Die Spiele laufen auf den CPU-Kernen,
die GPU trainiert nur die Netze.

Geschätzte Laufzeit für alle 10 Runden plus Ausnutzer-Tests: **1–2 Tage**. Die
Zahl ist eine Schätzung; nach Runde 4 zeigt `status.sh` die echte ETA.

## 2. Einmal einrichten

Per SSH verbinden, dann:

```bash
curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/gto/scripts/server_setup.sh | bash
cd ~/SkullKI
```

Das installiert alles, baut das Projekt, führt die Tests aus und prüft, ob
die GPU erkannt wird (`CUDA available: True`).

## 3. Starten

```bash
bash scripts/start.sh
```

Das Training läuft in einer **tmux-Sitzung im Hintergrund**. Du kannst das
Terminal schließen, die SSH-Verbindung trennen oder `Ctrl+C` in der
Statusanzeige drücken: Das Training läuft weiter.

## 4. Fortschritt ansehen

```bash
bash scripts/status.sh --watch     # Übersicht, aktualisiert alle 10 s
```

Beispiel:

```
state: RUNNING   current step: round 5: training
round  training                                  exploit seat1  exploit seat4
    1  done                                       +0.012±0.010   +0.009±0.010
    5  [######..........]  37.5% 15/40 ETA 1h12m              -              -
```

- **training**: Fortschrittsbalken, Iteration und Restzeit der Runde.
- **exploit seatX**: Punkte pro Runde, die ein trainierter Ausnutzer auf diesem
  Sitz gegen unsere Strategie gewinnt. **Je näher an 0, desto näher an GTO.**
  Der Wert ±… ist die Messungenauigkeit.

`Ctrl+C` schließt nur die Anzeige. Das rohe Live-Log siehst du mit
`tmux attach -t skullki`, verlassen ohne zu stoppen mit **`Ctrl+B`, dann `D`**.

## 5. Stoppen / Fortsetzen

```bash
bash scripts/stop.sh     # stoppt
bash scripts/start.sh    # setzt fort (fertige Runden werden nicht wiederholt)
```

Alle 5 Iterationen wird ein Zwischenstand gespeichert. Auch nach einem Absturz
oder Neustart des Servers einfach `start.sh` erneut ausführen.

Nur bestimmte Runden: `bash scripts/start.sh --rounds 4-10`.

## 6. Ergebnisse sichern

Alles liegt unter `~/SkullKI/runs/all/`:
- `roundNN/train/avg.bin` ist die fertige Strategie der Runde (das ist das Ergebnis),
- `summary.json` enthält die Ausnutzer-Messungen aller Runden,
- `run.log` / `console.log` sind die vollständigen Logs.

Vor dem Löschen des Servers herunterladen (auf deinem PC ausführen):

```bash
scp -r -P <port> root@<server>:~/SkullKI/runs/all ./runs_server
```

Die Strategie-Dateien sind klein (je ~5 MB). Die Checkpoints in `ckpt/` sind
groß und werden nur zum Fortsetzen gebraucht.

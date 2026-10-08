# Multiplayer-Server einrichten (dein Linux-Server)

Der Server ist ein kleines Node.js-Programm (`server/index.mjs`). Es mischt die
Karten, schickt jedem nur seine eigenen Karten und lässt die KI für freie
Plätze spielen. Dafür nutzt es dieselbe C++-Spiel-Engine wie die App. Es
braucht wenig: ca. 60 MB RAM, kaum CPU.

## 1. Einmalig installieren (auf dem Server, per SSH)

```bash
# Node.js 20 oder neuer (prüfen: node -v). Falls zu alt:
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs

git clone https://github.com/jAc0bll/SkullKI.git ~/SkullKI
cd ~/SkullKI && git checkout gto
cd server && npm ci --omit=dev

# Test: startet den Server, Strg+C beendet ihn
PORT=8787 node index.mjs
```

## 2. Dauerhaft laufen lassen (startet auch nach Neustart)

```bash
sudo cp ~/SkullKI/server/skullki.service /etc/systemd/system/
# Falls dein Benutzer nicht "jacob" heißt: User= und WorkingDirectory= in der Datei anpassen
sudo systemctl daemon-reload
sudo systemctl enable --now skullki
curl http://localhost:8787        # -> "SkullKI server ok"
```

## 3. Von außen erreichbar machen (Caddy, HTTPS)

1. **Adresse anlegen:** bei No-IP einen Host `skullki.schwartihost.com` auf
   dieselbe IP wie `24h.schwartihost.com` (deine öffentliche IP).
2. **Caddy:** in `/etc/caddy/Caddyfile` ergänzen:

   ```
   skullki.schwartihost.com {
       reverse_proxy localhost:8787
   }
   ```

   dann `sudo systemctl reload caddy`. Caddy holt das HTTPS-Zertifikat selbst.
   Die Weiterleitungen 80/443 im Router hast du für die andere Seite schon.

**Ohne neue Adresse geht es auch:** In den bestehenden Block von
`24h.schwartihost.com` schreiben

```
handle_path /skullki* {
    reverse_proxy localhost:8787
}
```

und in der App als Server `wss://24h.schwartihost.com/skullki` eintragen
(Spielen → Mit Freunden → „Server“ antippen).

Prüfen: <https://skullki.schwartihost.com> im Browser zeigt „SkullKI server ok“.

## Updates einspielen

```bash
cd ~/SkullKI && git pull && sudo systemctl restart skullki
```

Logs: `journalctl -u skullki -f`

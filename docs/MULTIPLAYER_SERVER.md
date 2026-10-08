# SkullKI auf deinen Linux-Server bringen

Danach öffnet jeder **https://skullki.schwartihost.com** im Handy-Browser und
kann sofort spielen: Solver, gegen die KI und mit Freunden. Ohne App Store und
ohne Sideloading. Über „Teilen → Zum Home-Bildschirm“ (iPhone) bzw.
„Zum Startbildschirm hinzufügen“ (Android) liegt SkullKI dann mit Icon wie eine
App auf dem Handy.

Dein Linux-Server macht dabei beides: die Website ausliefern und die
Multiplayer-Spiele leiten. Die KI-Bots für „gegen die KI“ rechnen auf dem
Handy selbst.

---

## Schritt 1: Mit dem Server verbinden

Auf deinem Windows-PC (gleiches WLAN wie der Server) die **Eingabeaufforderung**
öffnen (Windows-Taste, `cmd` tippen, Enter) und eingeben:

```
ssh jacob@192.168.178.189
```

Passwort eingeben (man sieht beim Tippen nichts, das ist normal), Enter.
Jetzt bist du auf dem Server.

## Schritt 2: SkullKI installieren (ein Befehl)

```
curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/gto/server/setup.sh | bash
```

Das Skript installiert Node.js (falls nötig), lädt SkullKI, baut die Website
und richtet einen Dienst ein, der auch nach einem Neustart von selbst startet.
Zwischendurch fragt es evtl. nach deinem Passwort (für `sudo`). Dauert ca.
3–5 Minuten. Am Ende steht:

```
SkullKI server ok · 0 Räume
Fertig. SkullKI läuft auf Port 8787.
```

**Test im Heimnetz:** auf dem Handy (im WLAN) `http://192.168.178.189:8787`
öffnen. Die App sollte erscheinen.

## Schritt 3: Adresse bei No-IP anlegen

1. Bei <https://www.noip.com> einloggen → **DDNS & Remote Access** → **Hostnames**.
2. **Create Hostname**:
   - Hostname: `skullki`, Domain: `schwartihost.com`
   - Typ: **A**, IPv4: **dieselbe IP wie bei `24h`** (deine öffentliche IP, z. B. die,
     die <https://ifconfig.me> anzeigt)
3. Speichern. Kann ein paar Minuten dauern, bis die Adresse überall bekannt ist.

## Schritt 4: Caddy die neue Adresse beibringen

Auf dem Server (Schritt 1):

```
sudo nano /etc/caddy/Caddyfile
```

Ganz unten diese drei Zeilen einfügen:

```
skullki.schwartihost.com {
    reverse_proxy localhost:8787
}
```

Speichern: **Strg+O**, **Enter**, dann beenden mit **Strg+X**. Dann:

```
sudo systemctl reload caddy
```

Caddy holt sich das HTTPS-Zertifikat automatisch. Die Weiterleitung der Ports
80/443 im Router auf den Server hast du für die 24h-Seite schon. Das gilt
auch für die neue Adresse.

## Schritt 5: Testen

Auf dem Handy **im Mobilfunk** (WLAN aus) <https://skullki.schwartihost.com>
öffnen. Dann Teilen → **Zum Home-Bildschirm**.

---

## Später: Updates einspielen

Wenn ich etwas Neues gebaut habe, einfach Schritt 1 und Schritt 2 wiederholen.
Derselbe Befehl aktualisiert alles.

## Wenn etwas nicht geht

| Problem | Lösung |
|---|---|
| Schritt 2 bricht ab | Die letzten Zeilen der Ausgabe als Screenshot schicken |
| Seite lädt nicht von außen, aber im WLAN (`:8787`) geht es | No-IP-Eintrag (Schritt 3) oder Caddy (Schritt 4) prüfen: `sudo journalctl -u caddy -n 30` |
| Dienst läuft? | `systemctl status skullki` |
| Log ansehen | `journalctl -u skullki -n 50` |

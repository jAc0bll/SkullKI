# SkullKI aufs iPhone (ohne App Store)

Die App wird automatisch in der Cloud gebaut (GitHub Actions, auf einem Mac).
Heraus kommt eine Datei `SkullKI.ipa`. Die spielst du mit **Sideloadly** von
deinem Windows-PC aufs iPhone. Sideloadly unterschreibt die App dabei mit deiner
Apple-ID. Das ist der offizielle Weg, wie Entwickler eigene Apps testen.

## Einmalig vorbereiten

1. **Sideloadly** installieren: <https://sideloadly.io> (Windows). Es braucht
   iTunes und iCloud von Apple. Sideloadly sagt dir beim Start, falls etwas fehlt.
2. iPhone per **USB-Kabel** anschließen und am iPhone „Vertrauen“ tippen.
3. Am iPhone **Entwicklermodus** einschalten: Einstellungen → Datenschutz &
   Sicherheit → Entwicklermodus → an → Neustart. (Der Punkt erscheint manchmal
   erst nach dem ersten Installationsversuch.)

## App holen und installieren

1. Auf GitHub: Repo → **Actions** → „iOS app“ → oberster grüner Lauf →
   unten bei **Artifacts** auf `SkullKI-ipa` klicken (Zip mit `SkullKI.ipa` drin, entpacken).
   Einen neuen Build startest du dort mit „Run workflow“. Er läuft auch
   automatisch, wenn sich der App-Code ändert, und dauert ca. 15–25 Minuten.
2. Sideloadly öffnen, `SkullKI.ipa` ins Fenster ziehen, deine Apple-ID
   eintragen, **Start**.
   - Tipp: Viele nehmen dafür eine zweite, kostenlose Apple-ID statt ihrer Haupt-ID.
3. Am iPhone: Einstellungen → Allgemein → **VPN & Geräteverwaltung** → deine
   Apple-ID → „Vertrauen“.
4. Fertig: SkullKI liegt auf dem Home-Bildschirm.

## Wie lange hält die App?

| Apple-ID | Laufzeit | Danach |
|---|---|---|
| kostenlos | **7 Tage** | in Sideloadly erneut „Start“ (Daten bleiben). Mit „Auto-Refresh“ macht Sideloadly das selbst, wenn PC und iPhone im selben WLAN sind |
| Apple-Entwicklerkonto (99 €/Jahr) | 1 Jahr | wie oben, einmal im Jahr |

Alternative zu Sideloadly: **AltStore** (<https://altstore.io>) mit AltServer auf
dem PC. Das erneuert die App automatisch im Hintergrund.

## Liquid Glass

Das echte Liquid-Glass-Material gibt es ab **iOS 26**. Auf älteren iPhones
(ab iOS 16.4) läuft die App auch, dann mit einer Milchglas-Nachbildung.

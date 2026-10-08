#!/usr/bin/env bash
# Installs or updates SkullKI on a Linux server (Debian/Ubuntu): the web app
# and the multiplayer server, as a service that also starts after a reboot.
# Run it again to update.
#
#   curl -fsSL https://raw.githubusercontent.com/jAc0bll/SkullKI/gto/server/setup.sh | bash
#
# Afterwards: http://localhost:8787 serves the app; put Caddy in front for
# HTTPS (docs/MULTIPLAYER_SERVER.md).
set -euo pipefail
DIR=${SKULLKI_DIR:-$HOME/SkullKI}
PORT=${PORT:-8787}

echo "== 1/5 Node.js und git"
if ! command -v node >/dev/null || [ "$(node -p 'process.versions.node.split(".")[0]')" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
command -v git >/dev/null || sudo apt-get install -y git
echo "   node $(node -v)"

echo "== 2/5 Code holen ($DIR)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q origin gto
  git -C "$DIR" checkout -q gto
  git -C "$DIR" reset -q --hard origin/gto
else
  git clone -q -b gto https://github.com/jAc0bll/SkullKI.git "$DIR"
fi

echo "== 3/5 Web-App bauen (1-3 Minuten)"
cd "$DIR/app"
npm ci --no-audit --no-fund --loglevel=error
npx expo export -p web --output-dir dist > /tmp/skullki-export.log 2>&1 || { tail -30 /tmp/skullki-export.log; exit 1; }

echo "== 4/5 Server einrichten"
cd "$DIR/server"
npm ci --omit=dev --no-audit --no-fund --loglevel=error

echo "== 5/5 Dienst starten"
sudo tee /etc/systemd/system/skullki.service > /dev/null <<UNIT
[Unit]
Description=SkullKI (web app + multiplayer)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR/server
Environment=PORT=$PORT
ExecStart=$(command -v node) index.mjs
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable -q skullki
sudo systemctl restart skullki
sleep 3
if curl -fsS "http://localhost:$PORT/health"; then
  echo
  echo "Fertig. SkullKI läuft auf Port $PORT."
  echo "Im Heimnetz testen: http://$(hostname -I | awk '{print $1}'):$PORT"
else
  echo "Der Dienst startet nicht. Log: journalctl -u skullki -n 50"
  exit 1
fi

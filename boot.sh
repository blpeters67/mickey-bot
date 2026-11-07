#!/bin/bash
# boot.sh — start the bot after the server boots (run from cron's @reboot).
#
# Install:  (crontab -l 2>/dev/null; echo "@reboot $(pwd)/boot.sh") | crontab -
# Log:      ~/swagbot-boot.log
#
# Waits for the network first: right after a power cut the router often comes
# back later than the server, and a bot started without DNS dies on login.
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$HOME/swagbot-boot.log"
exec >>"$LOG" 2>&1
echo "── $(date '+%F %T') boot start"
for i in $(seq 1 60); do                      # up to 5 minutes
    getent hosts discord.com >/dev/null && break
    sleep 5
done
getent hosts discord.com >/dev/null || echo "network still down after 5 min — starting anyway"
cd "$DIR" && ./start.sh

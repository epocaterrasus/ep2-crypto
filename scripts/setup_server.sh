#!/bin/bash
# One-time server setup for ep2-crypto on Hetzner
# Usage: ssh root@your-server < scripts/setup_server.sh
set -euo pipefail

echo "=== ep2-crypto server setup ==="

# 1. System updates
apt-get update && apt-get upgrade -y

# 2. Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed"
fi

# 3. Install Docker Compose plugin (v2)
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose..."
    apt-get install -y docker-compose-plugin
else
    echo "Docker Compose already installed"
fi

# 4. Install Doppler CLI
if ! command -v doppler &> /dev/null; then
    echo "Installing Doppler..."
    apt-get install -y apt-transport-https ca-certificates curl gnupg
    curl -sLf --retry 3 --tlsv1.2 --proto "=https" \
        'https://packages.doppler.com/public/cli/gpg.DE2A7741A397C129.key' | \
        gpg --dearmor -o /usr/share/keyrings/doppler-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/doppler-archive-keyring.gpg] https://packages.doppler.com/public/cli/deb/debian any-version main" | \
        tee /etc/apt/sources.list.d/doppler-cli.list
    apt-get update && apt-get install -y doppler
else
    echo "Doppler already installed"
fi

# 5. Install git
apt-get install -y git

# 6. Create deploy user
if ! id -u ep2 &> /dev/null; then
    echo "Creating ep2 user..."
    useradd -m -s /bin/bash -G docker ep2
else
    echo "ep2 user already exists"
fi

# 7. Clone repo
DEPLOY_DIR="/opt/ep2-crypto"
if [ ! -d "$DEPLOY_DIR" ]; then
    echo "Cloning repo..."
    git clone https://github.com/epocaterrasus/ep2-crypto.git "$DEPLOY_DIR"
    chown -R ep2:ep2 "$DEPLOY_DIR"
else
    echo "Repo already cloned at $DEPLOY_DIR"
fi

# 8. Create data directory
mkdir -p "$DEPLOY_DIR/data"
chown -R ep2:ep2 "$DEPLOY_DIR/data"

# 9. Firewall (UFW)
if command -v ufw &> /dev/null; then
    echo "Configuring firewall..."
    ufw allow 22/tcp    # SSH
    ufw allow 8000/tcp  # FastAPI
    ufw allow 3000/tcp  # Grafana
    ufw --force enable
else
    apt-get install -y ufw
    ufw allow 22/tcp
    ufw allow 8000/tcp
    ufw allow 3000/tcp
    ufw --force enable
fi

# 10. Systemd service for auto-start on boot
cat > /etc/systemd/system/ep2-crypto.service << 'EOF'
[Unit]
Description=ep2-crypto trading system
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/ep2-crypto
ExecStart=/usr/bin/docker compose -f docker/docker-compose.yml up -d
ExecStop=/usr/bin/docker compose -f docker/docker-compose.yml down
User=ep2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ep2-crypto

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Add SSH key for GitHub Actions:"
echo "     ssh-keygen -t ed25519 -f ~/.ssh/ep2_deploy -N ''"
echo "     cat ~/.ssh/ep2_deploy.pub >> /home/ep2/.ssh/authorized_keys"
echo "     cat ~/.ssh/ep2_deploy  # <- add this as HETZNER_SSH_KEY in GitHub secrets"
echo ""
echo "  2. Configure Doppler on the server:"
echo "     doppler login"
echo "     # Or set DOPPLER_TOKEN in /opt/ep2-crypto/docker/.env"
echo ""
echo "  3. Create Doppler service token:"
echo "     doppler configs tokens create --project ep2-crypto --config prd deploy-token"
echo "     # Save the token in /opt/ep2-crypto/docker/.env as DOPPLER_TOKEN=dp.st.xxx"
echo ""
echo "  4. Copy secrets to prd config:"
echo "     doppler secrets download --project ep2-crypto --config dev --no-file | doppler secrets upload --project ep2-crypto --config prd"
echo ""
echo "  5. Start the stack:"
echo "     cd /opt/ep2-crypto && docker compose -f docker/docker-compose.yml up -d"

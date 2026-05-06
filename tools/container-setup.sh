#!/bin/bash

# container-setup.sh
# Sets up a fresh container: fetches code, installs packages, etc.
#
# You must run this script *on the container*, not locally. To do so:
#
#     ssh root@REMOTE_HOST -T bash < container-setup.sh
#
#     Note: The above command will only work if you `ssh root`, not `ssh username`.
#
# Or, put it in a gist, ssh into the container, and remotely run:
#
#     curl -fsSL https://gist... | bash
#
# To set up a *local* Docker container:
#
#     docker run -it -v $(pwd)/tools/container-setup.sh:/container-setup.sh -e DEPLOY_PRIVATE_KEY="$(< ~/.ssh/deploy_private_key)" --gpus all pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime bash -c 'bash /container-setup.sh; exec bash'
#
# Before running, set your GIT_REPO and configure (below).
# Of course, customize to taste.

############################################### CONFIGURATION

# The git repo to pull your code from
GIT_REPO="TODO The repo where *your* code is. It will be automatically cloned to the container."

# If you're git repo requires an ssh key, you can:
# 1. Set DEPLOY_PRIVATE_KEY=<private ssh key> in this file
# 2. or set $DEPLOY_PRIVATE_KEY using your container's secrets interface
# 3. or copy the key beforehannd to /root/deploy_private_key

# Uncomment this to do apt-get upgrade (which can be slow)
APT_UPGRADE=true
#APT_UPGRADE_SECURITY_ONLY=true

# Packages to install
PKGS="git build-essential openssh-client sudo vim curl wget less nano bash-completion tmux htop bat"

# Files listed here will be persisted from  /home/clouduser/filename to /workspace/home_persistent/filename
PERSIST_THESE_HOME_FILES=".bash_history"

###############################################################

set -euo pipefail
SETUP_TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo -e "\n[*] Starting setup at $SETUP_TIMESTAMP"
echo "Container setup at $SETUP_TIMESTAMP" >> /etc/container_setup_at_time
set -x

echo -e "\n[*] Installing system packages"
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $PKGS
if [ -n "${APT_UPGRADE_SECURITY_ONLY:-}" ]; then
  echo -e "\n[*] Running apt-get upgrade security patches only (in background)"
  setsid sh -c '
  export DEBIAN_FRONTEND=noninteractive
  time \
    apt list --upgradable 2>/dev/null \
      | grep "/security" \
      | cut -d/ -f1 \
      | xargs -r apt-get install --only-upgrade -y \
          -o Dpkg::Options::="--force-confdef" \
          -o Dpkg::Options::="--force-confold"
' >> /root/upgrade.log 2>&1 &
fi
if [ -n "${APT_UPGRADE:-}" ]; then
  echo -e "\n[*] Running apt-get upgrade (in background)"
  setsid sh -c '
  export DEBIAN_FRONTEND=noninteractive
  time \
    apt list --upgradable 2>/dev/null \
      | grep -E "/(security|updates)" \
      | cut -d/ -f1 \
      | xargs -r apt-get install --only-upgrade -y \
          -o Dpkg::Options::="--force-confdef" \
          -o Dpkg::Options::="--force-confold"
' >> /root/upgrade.log 2>&1 &
fi
if command -v batcat &>/dev/null && ! command -v bat &>/dev/null; then
  ln -s "$(command -v batcat)" /usr/local/bin/bat
fi

#rm -rf /var/lib/apt/lists/*

echo -e "\n[*] Creating clouduser"
if ! id clouduser &>/dev/null; then
  adduser --disabled-password --gecos "" --shell /bin/bash clouduser
fi

echo -e "\n[*] Granting sudo to clouduser"
echo "clouduser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/clouduser
chmod 440 /etc/sudoers.d/clouduser

echo -e "\n[*] Setting up SSH keys for clouduser"
mkdir -p /home/clouduser/.ssh
chmod 700 /home/clouduser/.ssh
if [ -f /root/.ssh/authorized_keys ]; then
  cp /root/.ssh/authorized_keys /home/clouduser/.ssh/authorized_keys
  chown clouduser:clouduser /home/clouduser/.ssh/authorized_keys
  chmod 600 /home/clouduser/.ssh/authorized_keys
fi
chown clouduser:clouduser /home/clouduser/.ssh

echo -e "\n[*] Locking down SSHD configuration"
if [ -f /etc/ssh/sshd_config ]; then
  sed -ri 's/^[#[:space:]]*PasswordAuthentication[[:space:]]+.*/PasswordAuthentication no/' /etc/ssh/sshd_config || echo "PasswordAuthentication no" >> /etc/ssh/sshd_config
  sed -ri 's/^[#[:space:]]*ChallengeResponseAuthentication[[:space:]]+.*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config || echo "ChallengeResponseAuthentication no" >> /etc/ssh/sshd_config
  sed -ri 's/^[#[:space:]]*UsePAM[[:space:]]+.*/UsePAM yes/' /etc/ssh/sshd_config || echo "UsePAM yes" >> /etc/ssh/sshd_config
  sed -ri 's/^[#[:space:]]*PermitRootLogin[[:space:]]+.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config || echo "PermitRootLogin prohibit-password" >> /etc/ssh/sshd_config
fi

echo -e "\n[*] Installing deploy key"
if [ -n "${DEPLOY_PRIVATE_KEY:-}" ]; then
  echo -e "$DEPLOY_PRIVATE_KEY" > /home/clouduser/.ssh/id_ed25519
elif [ -f /root/deploy_private_key ]; then
  cp /root/deploy_private_key /home/clouduser/.ssh/id_ed25519
else
  echo -e "\n[!] No deploy key found"
fi
chmod 600 /home/clouduser/.ssh/id_ed25519 2>/dev/null || true
chown clouduser:clouduser /home/clouduser/.ssh/id_ed25519 2>/dev/null || true

echo -e "\n[*] Adding github.com to known_hosts"
if ! grep -q "github.com" /home/clouduser/.ssh/known_hosts 2>/dev/null; then
  su - clouduser -c "ssh-keyscan github.com >> ~/.ssh/known_hosts || true"
  chown clouduser:clouduser /home/clouduser/.ssh/known_hosts
  chmod 644 /home/clouduser/.ssh/known_hosts
fi

echo -e "\n[*] Setting clouduser PATH"
# Needed on some containers which give non-root users a deficient path
ROOT_PATH="$PATH"
cat > /etc/profile.d/clouduser-path.sh <<EOF
if [ "\$(id -u)" -eq "$(id -u clouduser)" ]; then
  export PATH="$ROOT_PATH:\$PATH"
fi
EOF
chmod 644 /etc/profile.d/clouduser-path.sh

echo -e "\n[*] Symlinking persistent home files"
for file in $PERSIST_THESE_HOME_FILES; do
  rm -f /home/clouduser/$file
  ln -s /workspace/home-persistent/$file /home/clouduser/$file
  chown -h clouduser:clouduser /home/clouduser/$file
done

echo -e "\n[*] Creating workspace directories"
mkdir -p /workspace/huggingface /workspace/home-persistent /workspace/repo
chown -R clouduser:clouduser /workspace || chmod a+rwx /workspace

echo -e "\n[*] Setting HF_HOME for all users"
echo "export HF_HOME=/workspace/huggingface" > /etc/profile.d/hf_home.sh
chmod 644 /etc/profile.d/hf_home.sh

REPO_DIR="/workspace/repo"
echo -e "\n[*] Fetching repository"
if [ -d "$REPO_DIR/.git" ]; then
  su - clouduser -c "cd $REPO_DIR && git pull"
else
  su - clouduser -c "git clone \"$GIT_REPO\" \"$REPO_DIR\""
  chown -R clouduser:clouduser "$REPO_DIR" 2>/dev/null || chmod a+rwx "$REPO_DIR"
fi

echo -e "\n[*] Creating ~/.tmux.conf"
cat > /home/clouduser/.tmux.conf <<EOF
set -g default-terminal "screen-256color"
set -as terminal-overrides ',xterm-256color:Tc'
set -g history-limit 10000
set -g status off
EOF
chown clouduser:clouduser /home/clouduser/.tmux.conf
chmod 644 /home/clouduser/.tmux.conf

echo -e "\n[*] Appending auto-cd to ~/.bashrc"
if ! grep -q "AUTO-CD TO REPO" /home/clouduser/.bashrc 2>/dev/null; then
  sudo -u clouduser bash -c "tee -a /home/clouduser/.bashrc << 'EOF'
# >> AUTO-CD TO REPO <<
if [[ \$- == *i* ]]; then
  cd $REPO_DIR
fi
# >> END AUTO-CD TO REPO <<
EOF"
fi

echo -e "\n[*] Appending tmux auto-attach to ~/.bashrc"
if ! grep -q "TMUX ATTACH" /home/clouduser/.bashrc 2>/dev/null; then
  sudo -u clouduser bash -c "tee -a /home/clouduser/.bashrc << 'EOF'
# >> TMUX ATTACH <<
if [[ -n \"\$SSH_CONNECTION\" && -z \"\$TMUX\" ]]; then
  echo -e \"\\nAttaching to tmux session in 3 seconds...\\nHit Ctrl-B then d to detach\"
  sleep 3
  tmux attach -t defaultsession 2>/dev/null || tmux new -s defaultsession
fi
# >> END TMUX ATTACH <<
EOF"
fi

echo -e "\n[*] Setting persistent ulimit for clouduser"
if ! grep -q "ULIMIT SET" /home/clouduser/.bashrc 2>/dev/null; then
  sudo -u clouduser bash -c "tee -a /home/clouduser/.bashrc << 'EOF'
# >> ULIMIT SET <<
ulimit -n 65536 || true
# >> END ULIMIT SET <<
EOF"
fi

echo -e "\n[*] Installing uv"
pip install --no-cache-dir uv

echo -e "\n[*] Installing Python dependencies"
cd "$REPO_DIR" && uv pip install --system -r requirements-external.txt
# If need be, you can just use pip
#cd $REPO_DIR && pip3 install -r requirements-external.txt
uv pip install --system pytest

echo -e "\n[*] Installing project"
cd "$REPO_DIR" && uv pip install --system -e .

echo -e "\n[*] Running pytest"
TEST_TIMESTAMP="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
if su - clouduser -c "cd $REPO_DIR && set -o pipefail && pytest --maxfail=1 --disable-warnings -q | tee /workspace/pytest.log"; then
  TEST_STATUS="PASSED"
else
  TEST_STATUS="FAILED"
fi
echo "pytest $TEST_STATUS at $TEST_TIMESTAMP" > /workspace/pytest.status

echo -e "\n[*] Updating motd"
sed -i '/pam_motd\.so[[:space:]]\+motd=\/run\/motd\.dynamic/s/^/#/' /etc/pam.d/login /etc/pam.d/sshd 2>/dev/null || true
cat > /etc/motd <<EOF

Container setup at: $SETUP_TIMESTAMP
Test status: $(cat /workspace/pytest.status 2>/dev/null || echo "Tests not run")
Use user 'clouduser' and cd to $REPO_DIR

EOF

echo "[*] Reloading sshd"
if pidof sshd &>/dev/null; then
  SSHD_MASTER=$(ps -C sshd -o pid=,ppid= | awk '$2==1 { print $1; exit }')
  if [ -n "$SSHD_MASTER" ]; then
    kill -HUP "$SSHD_MASTER" || true
  fi
fi

set +x
echo -e "\n\n[OK] SETUP COMPLETE at $(date -u '+%Y-%m-%d %H:%M:%S UTC') (started at $SETUP_TIMESTAMP)"
echo "You should now be able to connect as clouduser"

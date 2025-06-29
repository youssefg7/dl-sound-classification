
This works with any compute machine GCP/AWS/etc. 
For this guide I'll be using GCP e2-micro (free machine) and my m1 macbook.

## Locally
- Install gcp client
```bash
brew install --cask google-cloud-sdk
```

- Init gcp (login to your google account)
```bash
gcloud init
```

- ssh into the remote machine
```bash
gcloud compute ssh <username>@instance-<instance-id> --zone us-central1-c
```
## On Remote Machine
- Generate ssh key
```bash
ssh-keygen -t ed25519 -f ~/.ssh/ice_ed25519
ssh-copy-id -i ~/.ssh/ice_ed25519.pub <gatech-username>@login-ice.pace.gatech.edu`
```
- Add this config file
```sh
sudo nano ~/.ssh/config
```

```
Host ice
  HostName login-ice.pace.gatech.edu
  User yfouad3
  IdentityFile ~/.ssh/ice_ed25519
```
- Install vpn tools
```bash
sudo apt update && sudo apt install -y \
     openconnect tmux python3-pip
pip3 install --user vpn-slice
sudo ln -s $HOME/.local/bin/vpn-slice /usr/local/sbin/
```
- Create a one line bash file to startup the vpn
```bash
sudo nano ~/bin/gatech-vpn.sh
```

```bash
sudo openconnect --protocol=gp \
  --authgroup="DC Gateway" \
  --script "/usr/local/sbin/vpn-slice login-ice.pace.gatech.edu \
            128.61.0.0/16 130.207.0.0/16 10.0.0.0/8" \
  --background --pid-file /run/gatech-vpn.pid \
  vpn.gatech.edu
```

```bash
chmod +x ~/bin/gatech-vpn.sh
```

- Run the gatech vpn in a new tmux session 
```bash
tmux new -s gtvpn '~/bin/gatech-vpn.sh'
# enter GT password + Duo push, then detach (Ctrl-b d)
```

## Locally
- Setup ssh config
```
sudo nano ~/.ssh/config
```

```
Host gcp-jump
  HostName <remote machine public ip>
  User <remote machine username>
  IdentityFile ~/.ssh/google_compute_engine
  ForwardAgent yes
  ServerAliveInterval 60

Host ice
  HostName login-ice.pace.gatech.edu
  User <gatech-username>
  IdentityFile ~/.ssh/ice_ed25519
  ProxyJump gcp-jump
  ServerAliveInterval 60
```

- Test: `ssh ice`

## VS-Code Remote-SSH

1. Install **Remote – SSH**.
2. ⌘ ⇧ P → “Remote-SSH: Connect to Host…” → ice.
3. Status bar turns green → editing on ICE.

- Configure Github on ICE to use pull/push the repo remotely.
```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_ed25519 -C "ice-github"
cat ~/.ssh/github_ed25519.pub   # paste into GitHub → SSH keys
echo "Host github.com
  IdentityFile ~/.ssh/github_ed25519
  IdentitiesOnly yes" >> ~/.ssh/config
chmod 600 ~/.ssh/github_ed25519 ~/.ssh/config
git clone git@github.com:<org>/<repo>.git ~/dl-sound-classification
```

- Setup uv, venv, libs normally (inside /scratch folder)

- Request an interactive H100 session
```bash
salloc --qos=coc-ice \
       --gres=gpu:h100:1 \
       --time=08:00:00 \
       --cpus-per-task=8 --mem=64G \
```

- Access the interactive session
```bash
srun --pty bash -l 
```
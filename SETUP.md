# Vehicle Speed Detection Setup Guide

## Hardware
- Intel i7-4790K
- 32GB RAM
- No GPU required

## Folder Structure
This setup uses `/opt` for applications, keeping things clean for a multi-purpose server:
```
/opt/
├── speed-detection/        # This project (git repo)
│   ├── .git/               # Git repository data
│   ├── .gitignore          # Ignored files
│   ├── .env.example        # Example configuration template
│   ├── .env                # Your configuration (not tracked)
│   ├── README.md           # Project documentation
│   ├── SETUP.md            # Detailed setup guide
│   ├── requirements.txt    # Python dependencies
│   ├── speed_detect.py     # Main detection script
│   ├── config.json         # Calibration data (not tracked)
│   ├── venv/               # Python virtual environment (not tracked)
│   └── data/               # Logs, captured frames (not tracked)
│
└── databases/              # Future database data directories
```

## Part 1: Ubuntu Server 22.04 LTS Installation

### Create Bootable USB
1. Download Ubuntu Server 22.04 LTS from https://ubuntu.com/download/server
2. Use Rufus (Windows) or Balena Etcher to create a bootable USB

### Installation
1. Boot from USB, select "Install Ubuntu Server"
2. Choose language, keyboard layout
3. Select **"Ubuntu Server"** (standard, not minimized—you'll want the extra utilities)
4. Configure network (use DHCP or set static IP)
5. Configure storage:
   - Select **"Use an entire disk"** with **"Set up this disk as an LVM group"** enabled
   - This allows you to resize partitions later when you add database workloads
6. Configure swap: Set to **8-16GB** (helps with database memory spikes)
7. Create your user account
8. **Enable OpenSSH server** when prompted
9. Skip additional snaps
10. Reboot and remove USB

### Post-Install Setup
SSH in from your main machine:
```bash
ssh yourusername@SERVER_IP
```

Update the system:
```bash
sudo apt update && sudo apt upgrade -y
```

## Part 2: Install Dependencies

```bash
# Python and pip
sudo apt install -y python3 python3-pip python3-venv git

# OpenCV dependencies
sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

# Clone the repo
sudo mkdir -p /opt/speed-detection
sudo chown -R $USER:$USER /opt/speed-detection
git clone https://github.com/YOURUSERNAME/vehicle-speed-detection.git /opt/speed-detection
cd /opt/speed-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

## Part 3: Configure Environment

Copy the example environment file and edit it with your settings:

```bash
cd /opt/speed-detection
cp .env.example .env
nano .env
```

Update at minimum the `RTSP_URL` with your camera credentials:

```bash
# Required - your camera's RTSP stream
RTSP_URL=rtsp://admin:yourpassword@192.168.0.109:554/Streaming/Channels/2

# Optional - customize these as needed
WEB_PORT=8080
SPEED_WARNING=30
SPEED_DANGER=45
CONFIDENCE_THRESHOLD=0.5
PROCESS_EVERY_N_FRAMES=2
```

Use Channel 2 (substream) for lower resolution = faster processing.


## Part 4: Calibration (via Web UI)

Calibration is done through the web interface. No need to manually edit config files.

1. Run the script:
   ```bash
   cd /opt/speed-detection
   source venv/bin/activate
   python3 speed_detect.py
   ```

2. Open `http://SERVER_IP:8080` in your browser

3. Click **"Calibrate Now"**

4. Click 4 corners of a real-world rectangle (like the crosswalk):
   - Top-Left → Top-Right → Bottom-Right → Bottom-Left

5. Enter the real-world dimensions:
   - **Width**: distance perpendicular to the road (in feet)
   - **Height**: distance along the road (in feet)
   - Texas crosswalk stripes are typically 12" wide with 12-24" gaps

6. Click **Save**

The calibration is saved to `config.json` and persists across restarts.


## Part 5: Run the Detection

```bash
cd /opt/speed-detection
source venv/bin/activate
python3 speed_detect.py
```

### Command Line Options
```bash
# Standard operation with web interface
python3 speed_detect.py

# Open in browser: http://SERVER_IP:8080

# Save detection video for debugging
python3 speed_detect.py --save-video

# Override RTSP URL (optional, normally set in .env)
python3 speed_detect.py --rtsp "rtsp://user:pass@ip:554/path"
```

All other settings are configured in `.env`.

## Part 6: Run as a Service (Optional)

Create a systemd service to run at boot:

```bash
sudo nano /etc/systemd/system/speed-detection.service
```

Paste:
```ini
[Unit]
Description=Vehicle Speed Detection
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/opt/speed-detection
ExecStart=/opt/speed-detection/venv/bin/python3 speed_detect.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable speed-detection
sudo systemctl start speed-detection

# Check status
sudo systemctl status speed-detection

# View logs
journalctl -u speed-detection -f
```

## Troubleshooting

### RTSP Connection Issues
```bash
# Test with ffprobe
ffprobe "rtsp://admin:PASSWORD@192.168.0.109:554/Streaming/Channels/2"

# Check if port 554 is open
nc -zv 192.168.0.109 554
```

### Slow Performance
- Make sure you're using Channel 2 (substream)
- Reduce `PROCESS_EVERY_N_FRAMES` in the script
- Use YOLOv8n (nano) model - it's the default

### Speed Readings Seem Wrong
- Recalibrate via the web UI at `/calibrate`
- Double-check your real-world measurements
- Make sure you clicked the corners in the correct order
- The perspective transform is sensitive to accurate point placement
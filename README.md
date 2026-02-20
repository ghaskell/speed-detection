# Vehicle Speed & Traffic Detection System

A comprehensive vehicle monitoring system using YOLOv8 and OpenCV for real-time speed detection, traffic light monitoring, and violation capture. Designed for residential areas, school zones, and intersections.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### 🚗 Speed Detection
- Real-time vehicle speed calculation using perspective transform
- Multi-zone support for monitoring multiple lanes independently
- Automatic speeder capture with timestamped images
- Color-coded overlays (green/yellow/red based on speed)

### 🏫 School Zone Scheduling
- Configurable speed limits by day and time
- Automatic schedule switching (e.g., 20 mph during school hours, 30 mph otherwise)
- Visual indicator when school zone is active

### 🚦 Traffic Light Monitoring
- Automatic traffic light state detection (red/yellow/green) using HSV color analysis
- Red light runner detection
- Stop line violation detection
- Violation image capture with timestamps

### 📊 Dashboard & Analytics
- Real-time statistics (total vehicles, speeders, violations)
- Speeder rate percentage
- Heatmap showing when speeding occurs (day × hour)
- Recent detections feed

### 🌐 Web Interface
- Live video feed with overlays
- Zone calibration UI with click-to-define regions
- Traffic light configuration UI
- Schedule management
- Speeder and violation galleries

## Screenshots

| Live View | Dashboard | Calibration |
|-----------|-----------|-------------|
| Live feed with speed overlays | Statistics and heatmap | Click-to-define zones |

## Requirements

### Hardware
- Linux server (Ubuntu 22.04/24.04 recommended)
- IP camera with RTSP support (tested with Hikvision)
- Minimum 4GB RAM, 8GB+ recommended
- CPU with decent single-thread performance (or NVIDIA GPU for acceleration)

### Software
- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- Flask

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-speed-detection.git
cd vehicle-speed-detection
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
nano .env
```

Edit `.env` with your settings:

```env
# RTSP stream URL - use substream (Channel 2) for better performance
RTSP_URL=rtsp://username:password@192.168.1.100:554/Streaming/Channels/2

# Web server port
WEB_PORT=8080

# Default speed limit (mph) when no schedule is active
DEFAULT_SPEED_LIMIT=30

# Detection confidence threshold (0.0-1.0)
CONFIDENCE_THRESHOLD=0.5

# Process every Nth frame (higher = less CPU, lower accuracy)
PROCESS_EVERY_N_FRAMES=2
```

### 5. Run the Application

```bash
python speed_detect.py
```

Access the web interface at `http://your-server-ip:8080`

## Configuration

### Speed Zone Calibration

1. Navigate to `/calibrate`
2. For each lane you want to monitor:
   - Click 4 corners of a known rectangle (crosswalk stripes work well)
   - Order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**
   - Enter the real-world dimensions in feet
   - Click "Add Zone"

**Tips for accurate calibration:**
- Use crosswalk stripes (typically 12" wide with 12-24" gaps)
- Choose a rectangle that vehicles pass through
- Calibrate each lane separately for angled camera views

### Traffic Light Setup

1. Navigate to `/traffic_light`
2. **Light ROI**: Click two corners around the traffic light (top-left, bottom-right)
3. **Stop Line**: Click two points defining the stop line
4. **Intersection Zone**: Click 4 points defining the intersection area
5. Click "Save Configuration"

### School Zone Schedules

1. Navigate to `/schedules`
2. Set your default speed limit
3. Add schedules for reduced speed times:
   - Name (e.g., "Morning School Zone")
   - Speed limit (e.g., 20 mph)
   - Start/end times
   - Days of week

## Running as a Service

### Create systemd Service

```bash
sudo nano /etc/systemd/system/speed-detection.service
```

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
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable speed-detection
sudo systemctl start speed-detection
```

### View Logs

```bash
journalctl -u speed-detection -f
```

## Nginx Reverse Proxy (Optional)

For accessing via port 80 or adding SSL:

```bash
sudo nano /etc/nginx/sites-available/speed-detection
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/speed-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Web Interface Pages

| Route | Description |
|-------|-------------|
| `/` | Live video feed with speed overlays and current status |
| `/dashboard` | Statistics, heatmap, and recent detections |
| `/speeders` | Gallery of captured speeding vehicles |
| `/violations` | Gallery of traffic light violations |
| `/calibrate` | Speed zone calibration UI |
| `/traffic_light` | Traffic light and stop line configuration |
| `/schedules` | School zone schedule management |

## Data Storage

```
data/
├── speeders/          # Captured speeder images
│   └── 2024-01-15_14-30-45_42.5mph_Lane-1.jpg
├── violations/        # Traffic violation images
│   └── 2024-01-15_14-32-10_red_light_35.2mph.jpg
└── stats.json         # Statistics and heatmap data

config.json            # Zone and schedule configuration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | MJPEG video stream |
| `/calibration_frame` | GET | Current frame for calibration |
| `/add_zone` | POST | Add a speed detection zone |
| `/delete_zone` | POST | Remove a speed detection zone |
| `/save_traffic_light` | POST | Save traffic light configuration |
| `/add_schedule` | POST | Add a speed schedule |
| `/delete_schedule` | POST | Remove a speed schedule |

## Troubleshooting

### "Could not open RTSP stream"
- Verify camera IP and credentials
- Test stream with VLC: `vlc rtsp://user:pass@ip:554/path`
- Check firewall rules
- Try the substream (Channel 2) for lower bandwidth

### High CPU Usage
- Increase `PROCESS_EVERY_N_FRAMES` in `.env`
- Use the camera's substream instead of main stream
- Consider a machine with better single-thread CPU performance

### Inaccurate Speed Readings
- Recalibrate the zone with accurate real-world measurements
- Ensure the calibration rectangle is in the path vehicles travel
- Each lane needs its own zone for angled camera views

### Traffic Light Not Detecting
- Ensure the ROI is tightly around just the traffic light
- Check for sun glare washing out colors
- Adjust HSV thresholds in code if needed for your lighting conditions

### JSON Serialization Error (float32)
- This is fixed in the latest version
- Ensure speeds are converted: `round(float(speed), 1)`

## Performance Tuning

| Setting | Effect |
|---------|--------|
| `PROCESS_EVERY_N_FRAMES=1` | Highest accuracy, highest CPU |
| `PROCESS_EVERY_N_FRAMES=2` | Good balance (default) |
| `PROCESS_EVERY_N_FRAMES=3` | Lower CPU, may miss fast vehicles |
| `CONFIDENCE_THRESHOLD=0.3` | Detect more vehicles, more false positives |
| `CONFIDENCE_THRESHOLD=0.5` | Balanced (default) |
| `CONFIDENCE_THRESHOLD=0.7` | Fewer false positives, may miss some vehicles |

## Camera Recommendations

- **Resolution**: 720p substream is usually sufficient and reduces CPU load
- **Frame Rate**: 10-15 FPS is adequate for speed detection
- **Position**: Mount with clear view of road, ideally perpendicular to traffic flow
- **Angle**: Steeper angles require per-lane calibration zones

## How It Works

### Speed Calculation

1. **Vehicle Detection**: YOLOv8 identifies vehicles (cars, trucks, motorcycles, buses) in each frame
2. **Tracking**: ByteTrack maintains vehicle identities across frames
3. **Position Mapping**: A perspective transform converts pixel coordinates to real-world feet based on your calibration
4. **Speed Calculation**: Distance traveled ÷ time elapsed = speed in feet/second, converted to mph
5. **Smoothing**: Speeds are averaged over several frames to reduce noise

### Traffic Light Detection

1. **ROI Extraction**: The configured rectangle around the traffic light is extracted
2. **Color Analysis**: HSV color space is used to identify red, yellow, and green pixels
3. **State Determination**: The color with the most bright pixels determines the light state
4. **Violation Detection**: Vehicles crossing the stop line or entering the intersection during red are flagged

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] License plate recognition (ALPR)
- [ ] Push notifications (Pushover, Telegram, email)
- [ ] Video clip capture (before/after speeding event)
- [ ] Home Assistant integration
- [ ] Vehicle color detection
- [ ] Repeat offender tracking
- [ ] Public statistics dashboard
- [ ] GPU acceleration support
- [ ] Multi-camera support
- [ ] Database storage (SQLite/PostgreSQL)
- [ ] REST API for external integrations
- [ ] Mobile app

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision
- [Flask](https://flask.palletsprojects.com/) for the web interface

## Disclaimer

This system is intended for personal/educational use. Speed and violation data should not be used for legal enforcement purposes. Always comply with local laws regarding surveillance and data collection. The accuracy of speed measurements depends on proper calibration and camera positioning.

---

**Made with ❤️ for safer neighborhoods**
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vehicle speed and traffic violation detection system for a school zone intersection in San Antonio, TX. Monitors traffic via RTSP camera feed using YOLOv8 + ByteTrack, calculates vehicle speeds via perspective transform, detects red-light violations, and provides a web dashboard.

**Hardware**: Intel i7-4790K, 32GB RAM, no GPU. Camera: Hikvision at `<camera-ip>` (720p substream). Web UI at `http://<server-ip>:8080`.

## Running the Application

```bash
# Development
cd /opt/speed-detection
source venv/bin/activate
python speed_detect.py

# Run with explicit RTSP URL
python speed_detect.py --rtsp "rtsp://user:pass@ip:554/path"

# Production (systemd)
sudo systemctl start speed-detection
sudo systemctl status speed-detection
journalctl -u speed-detection -f
```

The YOLO model (`yolov8n.pt`) is downloaded automatically on first run. There is no build system, test suite, or linter configuration.

## Testing Changes

1. `sudo systemctl stop speed-detection`
2. `python speed_detect.py` (run manually, check console)
3. Verify at `http://<server-ip>:8080`
4. `sudo systemctl start speed-detection`

## Architecture

Everything lives in a single file: `speed_detect.py` (~1,500 lines). All Flask routes, HTML templates (as inline Python strings), detection logic, and config handling are there.

**Threading model**: Flask runs on a background daemon thread. The main thread runs the OpenCV/YOLO detection loop, writing annotated frames to global `output_frame` protected by `frame_lock`. Config and stats globals are shared between threads without additional locking.

**Code style**: Global variables for shared state, no type hints, section headers with `# ====` blocks.

### Key Components

- **`VehicleTracker` class** — Core tracking engine. Uses `cv2.getPerspectiveTransform` to map pixel coordinates to real-world feet via user-calibrated 4-point zones. Speed = Euclidean distance in feet/sec × 0.681818 (mph conversion), smoothed over 3 frames. Tracks keyed by YOLO track ID. Key methods: `update()`, `calculate_speed()`, `check_traffic_violations()`.

- **`detect_light_state(frame)`** — Extracts configured ROI, converts to HSV, counts red/yellow/green pixels (red uses two HSV ranges for hue wraparound). Updates global `current_light_state`.

- **`main(args)`** — Entry point: loads config/stats, starts Flask daemon thread, opens RTSP stream, runs frame loop with `model.track(..., tracker="bytetrack.yaml", persist=True)`, auto-reconnects on stream loss.

- **`get_current_speed_limit()`** — Checks current day/time against JSON-persisted schedules for school zone speed limit enforcement.

### YOLO Setup

- Model: `yolov8n` (nano, CPU-optimized for i7-4790K)
- Tracked vehicle classes: `[2, 3, 5, 7]` = car, motorcycle, bus, truck
- Frame skipping via `PROCESS_EVERY_N_FRAMES` for CPU headroom

### Runtime Config & Data (gitignored)

| Path | Purpose |
|---|---|
| `config.json` | Zone calibrations, traffic light ROI/stop line/intersection, schedules |
| `data/stats.json` | Vehicle counts, speeder counts, heatmap data, violations |
| `data/speeders/` | JPEG crops of speeding vehicles |
| `data/violations/` | JPEG crops of red-light violations |

### Configuration (.env)

- `RTSP_URL`, `WEB_PORT` (default: 8080), `DEFAULT_SPEED_LIMIT` (default: 30 mph)
- `CONFIDENCE_THRESHOLD` (default: 0.5), `PROCESS_EVERY_N_FRAMES` (default: 2)

### Web Routes

`/` live view, `/dashboard` stats/heatmap, `/video_feed` MJPEG stream, `/calibrate` zone setup, `/traffic_light` light config, `/schedules` school zone schedules, `/speeders` and `/violations` image galleries.

## Common Tasks

### Adding a new web page
1. Add template string: `NEW_TEMPLATE = """..."""`
2. Add route: `@app.route('/new')` returning `render_template_string(NEW_TEMPLATE, **context)`

### Adding new stats
1. Add field to `stats` dict in `load_stats()`
2. Update in `record_speed()` or `record_violation()`
3. Display in `DASHBOARD_TEMPLATE`

## Debugging

```bash
# Test RTSP stream
ffmpeg -i "rtsp://user:pass@ip:554/path" -frames:v 1 test.jpg

# Service logs
journalctl -u speed-detection -n 100
```

**Common errors**:
- `float32 not JSON serializable` — wrap numpy values: `float(value)`
- `Could not open RTSP stream` — check URL, credentials, network
- High CPU — increase `PROCESS_EVERY_N_FRAMES`, use camera substream

## Future Improvements

- Split into multiple files (routes.py, tracker.py, etc.)
- SQLite storage, push notifications, license plate recognition
- Video clip capture, Home Assistant integration, GPU acceleration

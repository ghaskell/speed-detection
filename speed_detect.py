#!/usr/bin/env python3
"""
Vehicle Speed Detection with Traffic Light Monitoring
Features: Multi-zone speed detection, school zone scheduling, red light runner detection, stop line violations
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import argparse
from datetime import datetime, timedelta
from threading import Thread, Lock
from flask import Flask, Response, render_template_string, request, jsonify
import json
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:PASSWORD@192.168.1.100:554/Streaming/Channels/2")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
DEFAULT_SPEED_LIMIT = int(os.getenv("DEFAULT_SPEED_LIMIT", "30"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))

CONFIG_FILE = "config.json"
STATS_FILE = "data/stats.json"
SPEEDERS_DIR = "data/speeders"
VIOLATIONS_DIR = "data/violations"
MIN_TRACK_LENGTH = 5
SPEED_SMOOTHING_WINDOW = 3
VEHICLE_CLASSES = [2, 3, 5, 7]

ZONE_COLORS = [(255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0)]

# =============================================================================
# GLOBALS
# =============================================================================

app = Flask(__name__)
output_frame = None
frame_lock = Lock()
calibration_frame = None
current_light_state = "unknown"  # red, yellow, green, unknown

config = {
    "zones": [],
    "speed_schedules": [
        {"name": "School Hours", "days": [1,2,3,4,5], "start": "07:30", "end": "08:30", "limit": 20},
        {"name": "School Hours", "days": [1,2,3,4,5], "start": "14:30", "end": "16:00", "limit": 20},
    ],
    "default_limit": DEFAULT_SPEED_LIMIT,
    "traffic_light": None,  # {"roi": [x1,y1,x2,y2], "stop_line": [[x1,y1],[x2,y2]], "intersection_zone": [[x1,y1],...]}
}

stats = {
    "total_vehicles": 0,
    "total_speeders": 0,
    "total_red_light_runners": 0,
    "total_stop_line_violations": 0,
    "speeds": [],
    "heatmap": {},
    "violations": []
}

# =============================================================================
# TRAFFIC LIGHT DETECTION
# =============================================================================

def detect_light_state(frame):
    """Detect traffic light state from ROI using color analysis"""
    global current_light_state
    
    tl_config = config.get("traffic_light")
    if not tl_config or not tl_config.get("roi"):
        return "unknown"
    
    roi = tl_config["roi"]
    x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    
    # Extract ROI
    light_roi = frame[y1:y2, x1:x2]
    if light_roi.size == 0:
        return "unknown"
    
    # Convert to HSV
    hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)
    
    # Define color ranges (HSV)
    # Red has two ranges (wraps around 180)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])
    
    # Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # Count pixels
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    # Determine state based on which has most bright pixels
    min_pixels = 50  # Minimum pixels to consider valid
    
    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > min_pixels:
        current_light_state = "red"
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > min_pixels:
        current_light_state = "yellow"
    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > min_pixels:
        current_light_state = "green"
    else:
        current_light_state = "unknown"
    
    return current_light_state

def point_past_line(point, line_start, line_end, direction="down"):
    """Check if a point has crossed the stop line
    direction: 'down' means vehicles moving down the frame, 'up' means moving up
    """
    # Using cross product to determine which side of the line the point is on
    # Line from A to B, point P
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    
    # Cross product (B-A) x (P-A)
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    
    if direction == "down":
        return cross > 0  # Point is below the line
    else:
        return cross < 0  # Point is above the line

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using cv2"""
    if not polygon or len(polygon) < 3:
        return False
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

# =============================================================================
# SPEED SCHEDULE FUNCTIONS
# =============================================================================

def get_current_speed_limit():
    now = datetime.now()
    current_day = now.isoweekday()
    current_time = now.strftime("%H:%M")
    
    for schedule in config.get("speed_schedules", []):
        if current_day in schedule.get("days", []):
            if schedule.get("start", "00:00") <= current_time <= schedule.get("end", "23:59"):
                return schedule.get("limit", DEFAULT_SPEED_LIMIT)
    
    return config.get("default_limit", DEFAULT_SPEED_LIMIT)

def get_schedule_status():
    limit = get_current_speed_limit()
    default = config.get("default_limit", DEFAULT_SPEED_LIMIT)
    if limit < default:
        return f"School Zone Active - {limit} mph"
    return f"Normal - {limit} mph"

# =============================================================================
# STATS FUNCTIONS
# =============================================================================

def load_stats():
    global stats
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            stats["speeds"] = [s for s in stats.get("speeds", []) if s.get("timestamp", "") > cutoff]
            stats["violations"] = [v for v in stats.get("violations", []) if v.get("timestamp", "") > cutoff]
        except:
            pass

def save_stats():
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

def record_speed(speed, zone_name, is_speeder=False):
    now = datetime.now()
    stats["total_vehicles"] = stats.get("total_vehicles", 0) + 1
    
    if is_speeder:
        stats["total_speeders"] = stats.get("total_speeders", 0) + 1
        key = f"{now.isoweekday()}_{now.hour}"
        if "heatmap" not in stats:
            stats["heatmap"] = {}
        stats["heatmap"][key] = stats["heatmap"].get(key, 0) + 1
    
    if "speeds" not in stats:
        stats["speeds"] = []
    stats["speeds"].append({
        "timestamp": now.isoformat(),
        "speed": round(float(speed), 1),
        "zone": zone_name,
        "limit": int(get_current_speed_limit()),
        "speeder": is_speeder
    })
    if len(stats["speeds"]) > 1000:
        stats["speeds"] = stats["speeds"][-1000:]
    
    if stats["total_vehicles"] % 10 == 0:
        save_stats()

def record_violation(violation_type, track_id, speed=None):
    now = datetime.now()
    
    if violation_type == "red_light":
        stats["total_red_light_runners"] = stats.get("total_red_light_runners", 0) + 1
    elif violation_type == "stop_line":
        stats["total_stop_line_violations"] = stats.get("total_stop_line_violations", 0) + 1
    
    if "violations" not in stats:
        stats["violations"] = []
    stats["violations"].append({
        "timestamp": now.isoformat(),
        "type": violation_type,
        "track_id": int(track_id),
        "speed": round(float(speed), 1) if speed else None
    })
    if len(stats["violations"]) > 500:
        stats["violations"] = stats["violations"][-500:]
    
    save_stats()

# =============================================================================
# CONFIG MANAGEMENT
# =============================================================================

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                if "source_points" in loaded and "zones" not in loaded:
                    config["zones"] = [{
                        "name": "Zone 1",
                        "source_points": loaded["source_points"],
                        "real_width": loaded.get("real_width", 24.0),
                        "real_height": loaded.get("real_height", 50.0)
                    }]
                else:
                    config.update(loaded)
            print(f"Loaded config from {CONFIG_FILE}")
        except Exception as e:
            print(f"Could not load config: {e}")

def save_config():
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {CONFIG_FILE}")

os.makedirs(SPEEDERS_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

# =============================================================================
# WEB TEMPLATES
# =============================================================================

MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speed & Traffic Detection</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 20px; }
        img.stream { width: 100%; border: 2px solid #333; border-radius: 8px; }
        .status-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 10px 15px; background: #2a2a2a; border-radius: 8px; flex-wrap: wrap; gap: 10px; }
        .status { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
        .badge { padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 13px; }
        .badge.green { background: #4ade80; color: #000; }
        .badge.yellow { background: #fbbf24; color: #000; }
        .badge.red { background: #f87171; color: #000; }
        .badge.gray { background: #4b5563; color: #fff; }
        .btn { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; text-decoration: none; }
        .btn:hover { background: #2563eb; }
        .btn.secondary { background: #4b5563; }
        .nav-links { display: flex; gap: 10px; flex-wrap: wrap; }
        .legend { margin-top: 15px; text-align: center; color: #888; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speed & Traffic Detection</h1>
        <div class="status-bar">
            <div class="status">
                <span class="badge {{ 'yellow' if 'School' in schedule_status else 'green' }}">{{ current_limit }} mph</span>
                <span class="badge {{ {'red':'red','yellow':'yellow','green':'green'}.get(light_state, 'gray') }}">Light: {{ light_state.upper() }}</span>
                <span style="color:#888">{{ zone_count }} Zone(s)</span>
            </div>
            <div class="nav-links">
                <a href="/dashboard" class="btn">Dashboard</a>
                <a href="/violations" class="btn secondary">Violations</a>
                <a href="/speeders" class="btn secondary">Speeders</a>
                <a href="/calibrate" class="btn secondary">Zones</a>
                <a href="/traffic_light" class="btn secondary">Traffic Light</a>
                <a href="/schedules" class="btn secondary">Schedule</a>
            </div>
        </div>
        <img class="stream" src="/video_feed" alt="Video Feed">
        <div class="legend">Limit: {{ current_limit }} mph | Light: {{ light_state }}</div>
    </div>
</body>
</html>
"""

TRAFFIC_LIGHT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Light Setup</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1, h2, h3 { text-align: center; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .instructions { background: #2a2a2a; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; }
        .calibration-area { display: flex; gap: 20px; flex-wrap: wrap; }
        .image-container { flex: 1; min-width: 600px; position: relative; }
        #calibration-image { width: 100%; border: 2px solid #333; border-radius: 8px; cursor: crosshair; }
        .sidebar { width: 320px; background: #2a2a2a; padding: 20px; border-radius: 8px; }
        .mode-buttons { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .mode-btn { flex: 1; padding: 10px; border: 2px solid #444; background: #1a1a1a; color: #888; border-radius: 6px; cursor: pointer; text-align: center; }
        .mode-btn.active { border-color: #3b82f6; color: #3b82f6; background: rgba(59,130,246,0.1); }
        .config-section { margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #333; }
        .config-section h4 { margin: 0 0 10px 0; color: #3b82f6; }
        .config-item { display: flex; justify-content: space-between; padding: 5px 0; color: #888; font-size: 13px; }
        .btn { background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; width: 100%; margin-bottom: 10px; }
        .btn:hover { background: #2563eb; }
        .btn.secondary { background: #4b5563; }
        .btn.danger { background: #dc2626; }
        .status-preview { padding: 15px; background: #1a1a1a; border-radius: 8px; text-align: center; margin-bottom: 15px; }
        .light-indicator { display: inline-block; width: 30px; height: 30px; border-radius: 50%; margin: 0 5px; }
        .light-indicator.red { background: {{ '#f87171' if light_state == 'red' else '#4b5563' }}; }
        .light-indicator.yellow { background: {{ '#fbbf24' if light_state == 'yellow' else '#4b5563' }}; }
        .light-indicator.green { background: {{ '#4ade80' if light_state == 'green' else '#4b5563' }}; }
        svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Traffic Light Setup</h1>
        <a href="/" class="back-link">← Back to Live View</a>
        
        <div class="instructions">
            <h3 style="margin-top:0;color:#3b82f6">Setup Instructions</h3>
            <ol>
                <li><strong>Traffic Light ROI:</strong> Click two corners (top-left, bottom-right) around the traffic light</li>
                <li><strong>Stop Line:</strong> Click two points defining the stop line</li>
                <li><strong>Intersection Zone:</strong> Click 4 points defining the intersection area (vehicles here during red = runners)</li>
            </ol>
        </div>
        
        <div class="calibration-area">
            <div class="image-container">
                <img id="calibration-image" src="/calibration_frame" alt="Frame">
                <svg id="overlay"></svg>
            </div>
            
            <div class="sidebar">
                <div class="status-preview">
                    <div>Current Light State</div>
                    <div style="margin-top:10px">
                        <span class="light-indicator red"></span>
                        <span class="light-indicator yellow"></span>
                        <span class="light-indicator green"></span>
                    </div>
                    <div style="margin-top:10px;font-size:20px;font-weight:bold">{{ light_state.upper() }}</div>
                </div>
                
                <div class="mode-buttons">
                    <button class="mode-btn active" data-mode="roi">Light ROI</button>
                    <button class="mode-btn" data-mode="stopline">Stop Line</button>
                    <button class="mode-btn" data-mode="intersection">Intersection</button>
                </div>
                
                <div class="config-section">
                    <h4>Traffic Light ROI</h4>
                    <div class="config-item"><span>Top-Left:</span><span id="roi-tl">{{ tl_config.roi[0:2] if tl_config and tl_config.roi else '-' }}</span></div>
                    <div class="config-item"><span>Bottom-Right:</span><span id="roi-br">{{ tl_config.roi[2:4] if tl_config and tl_config.roi else '-' }}</span></div>
                </div>
                
                <div class="config-section">
                    <h4>Stop Line</h4>
                    <div class="config-item"><span>Point 1:</span><span id="sl-p1">{{ tl_config.stop_line[0] if tl_config and tl_config.stop_line else '-' }}</span></div>
                    <div class="config-item"><span>Point 2:</span><span id="sl-p2">{{ tl_config.stop_line[1] if tl_config and tl_config.stop_line|length > 1 else '-' }}</span></div>
                </div>
                
                <div class="config-section">
                    <h4>Intersection Zone</h4>
                    <div class="config-item"><span>Points:</span><span id="iz-count">{{ tl_config.intersection_zone|length if tl_config and tl_config.intersection_zone else 0 }}/4</span></div>
                </div>
                
                <button class="btn" id="save-btn">Save Configuration</button>
                <button class="btn secondary" id="reset-btn">Reset Current Mode</button>
                <button class="btn danger" id="clear-btn">Clear All</button>
            </div>
        </div>
    </div>
    
    <script>
        const img = document.getElementById('calibration-image');
        const svg = document.getElementById('overlay');
        let mode = 'roi';
        let data = {
            roi: {{ (tl_config.roi if tl_config and tl_config.roi else []) | tojson }},
            stop_line: {{ (tl_config.stop_line if tl_config and tl_config.stop_line else []) | tojson }},
            intersection_zone: {{ (tl_config.intersection_zone if tl_config and tl_config.intersection_zone else []) | tojson }}
        };
        
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                mode = btn.dataset.mode;
            });
        });
        
        img.addEventListener('click', function(e) {
            const rect = img.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * img.naturalWidth / rect.width);
            const y = Math.round((e.clientY - rect.top) * img.naturalHeight / rect.height);
            
            if (mode === 'roi') {
                if (data.roi.length >= 4) data.roi = [];
                data.roi.push(x, y);
                if (data.roi.length === 2) {
                    document.getElementById('roi-tl').textContent = `[${data.roi[0]}, ${data.roi[1]}]`;
                } else if (data.roi.length === 4) {
                    document.getElementById('roi-br').textContent = `[${data.roi[2]}, ${data.roi[3]}]`;
                }
            } else if (mode === 'stopline') {
                if (data.stop_line.length >= 2) data.stop_line = [];
                data.stop_line.push([x, y]);
                if (data.stop_line.length === 1) {
                    document.getElementById('sl-p1').textContent = `[${x}, ${y}]`;
                } else {
                    document.getElementById('sl-p2').textContent = `[${x}, ${y}]`;
                }
            } else if (mode === 'intersection') {
                if (data.intersection_zone.length >= 4) data.intersection_zone = [];
                data.intersection_zone.push([x, y]);
                document.getElementById('iz-count').textContent = `${data.intersection_zone.length}/4`;
            }
            
            drawOverlay();
        });
        
        function drawOverlay() {
            const rect = img.getBoundingClientRect();
            const sx = rect.width / img.naturalWidth;
            const sy = rect.height / img.naturalHeight;
            let html = '';
            
            // Draw ROI
            if (data.roi.length === 4) {
                const [x1, y1, x2, y2] = data.roi;
                html += `<rect x="${x1*sx}" y="${y1*sy}" width="${(x2-x1)*sx}" height="${(y2-y1)*sy}" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" stroke-width="2"/>`;
            }
            
            // Draw stop line
            if (data.stop_line.length === 2) {
                const [p1, p2] = data.stop_line;
                html += `<line x1="${p1[0]*sx}" y1="${p1[1]*sy}" x2="${p2[0]*sx}" y2="${p2[1]*sy}" stroke="#f87171" stroke-width="3"/>`;
            }
            
            // Draw intersection zone
            if (data.intersection_zone.length >= 3) {
                let path = data.intersection_zone.map((p, i) => `${i===0?'M':'L'}${p[0]*sx},${p[1]*sy}`).join(' ');
                if (data.intersection_zone.length === 4) path += ' Z';
                html += `<path d="${path}" fill="rgba(251,191,36,0.2)" stroke="#fbbf24" stroke-width="2"/>`;
            }
            
            svg.innerHTML = html;
        }
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            if (mode === 'roi') { data.roi = []; document.getElementById('roi-tl').textContent = '-'; document.getElementById('roi-br').textContent = '-'; }
            else if (mode === 'stopline') { data.stop_line = []; document.getElementById('sl-p1').textContent = '-'; document.getElementById('sl-p2').textContent = '-'; }
            else { data.intersection_zone = []; document.getElementById('iz-count').textContent = '0/4'; }
            drawOverlay();
        });
        
        document.getElementById('clear-btn').addEventListener('click', () => {
            if (!confirm('Clear all traffic light configuration?')) return;
            fetch('/clear_traffic_light', { method: 'POST' }).then(() => location.reload());
        });
        
        document.getElementById('save-btn').addEventListener('click', () => {
            fetch('/save_traffic_light', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            }).then(r => r.json()).then(d => {
                if (d.success) alert('Saved!');
                else alert('Error saving');
            });
        });
        
        img.onload = drawOverlay;
        drawOverlay();
    </script>
</body>
</html>
"""

VIOLATIONS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Violations - Traffic Detection</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #2a2a2a; border-radius: 8px; padding: 20px; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; }
        .stat-value.red { color: #f87171; }
        .stat-value.yellow { color: #fbbf24; }
        .stat-label { color: #888; margin-top: 5px; font-size: 13px; }
        .violations-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .violation-card { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        .violation-card img { width: 100%; height: 180px; object-fit: cover; }
        .violation-info { padding: 15px; }
        .violation-type { font-size: 14px; font-weight: bold; padding: 3px 8px; border-radius: 4px; display: inline-block; }
        .violation-type.red_light { background: #f87171; color: #000; }
        .violation-type.stop_line { background: #fbbf24; color: #000; }
        .violation-meta { color: #888; font-size: 13px; margin-top: 8px; }
        .no-violations { text-align: center; color: #888; padding: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Traffic Violations</h1>
        <a href="/" class="back-link">← Back to Live View</a>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value red">{{ red_light_count }}</div>
                <div class="stat-label">Red Light Runners</div>
            </div>
            <div class="stat-card">
                <div class="stat-value yellow">{{ stop_line_count }}</div>
                <div class="stat-label">Stop Line Violations</div>
            </div>
        </div>
        
        {% if violations %}
        <div class="violations-grid">
            {% for v in violations %}
            <div class="violation-card">
                <img src="/violation_image/{{ v.filename }}" alt="Violation">
                <div class="violation-info">
                    <span class="violation-type {{ v.type }}">{{ 'Red Light' if v.type == 'red_light' else 'Stop Line' }}</span>
                    <div class="violation-meta">{{ v.time }}{% if v.speed %} · {{ v.speed }} mph{% endif %}</div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-violations">No violations captured yet</div>
        {% endif %}
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard - Speed Detection</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1, h2 { text-align: center; }
        h2 { margin-top: 30px; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #2a2a2a; border-radius: 8px; padding: 20px; text-align: center; }
        .stat-value { font-size: 28px; font-weight: bold; color: #3b82f6; }
        .stat-value.danger { color: #f87171; }
        .stat-value.warning { color: #fbbf24; }
        .stat-value.success { color: #4ade80; }
        .stat-label { color: #888; margin-top: 5px; font-size: 12px; }
        .heatmap-container { overflow-x: auto; }
        .heatmap { display: grid; grid-template-columns: 60px repeat(24, 1fr); gap: 2px; min-width: 700px; }
        .heatmap-cell { aspect-ratio: 1; border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 10px; }
        .heatmap-header, .heatmap-day { background: transparent; color: #666; font-size: 11px; }
        .heatmap-day { justify-content: flex-end; padding-right: 8px; }
        .heat-0 { background: #1e293b; } .heat-1 { background: #365314; } .heat-2 { background: #4d7c0f; }
        .heat-3 { background: #84cc16; } .heat-4 { background: #fbbf24; } .heat-5 { background: #f97316; } .heat-6 { background: #ef4444; }
        .recent-speeds { background: #2a2a2a; border-radius: 8px; overflow: hidden; max-height: 400px; overflow-y: auto; }
        .speed-row { display: flex; justify-content: space-between; padding: 10px 15px; border-bottom: 1px solid #333; font-size: 13px; }
        .speed-row.speeder { background: rgba(239, 68, 68, 0.1); }
        .speed-value { font-weight: bold; }
        .speed-value.over { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dashboard</h1>
        <a href="/" class="back-link">← Back to Live View</a>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{{ total_vehicles }}</div><div class="stat-label">Total Vehicles</div></div>
            <div class="stat-card"><div class="stat-value danger">{{ total_speeders }}</div><div class="stat-label">Speeders</div></div>
            <div class="stat-card"><div class="stat-value warning">{{ speeder_percent }}%</div><div class="stat-label">Speeder Rate</div></div>
            <div class="stat-card"><div class="stat-value">{{ avg_speed }}</div><div class="stat-label">Avg Speed</div></div>
            <div class="stat-card"><div class="stat-value danger">{{ max_speed }}</div><div class="stat-label">Max Speed</div></div>
            <div class="stat-card"><div class="stat-value danger">{{ red_light_runners }}</div><div class="stat-label">Red Light Runners</div></div>
            <div class="stat-card"><div class="stat-value warning">{{ stop_line_violations }}</div><div class="stat-label">Stop Line Violations</div></div>
            <div class="stat-card"><div class="stat-value success">{{ current_limit }}</div><div class="stat-label">Current Limit</div></div>
        </div>
        
        <h2>Speeder Heatmap</h2>
        <div class="heatmap-container">
            <div class="heatmap">
                <div class="heatmap-cell heatmap-header"></div>
                {% for h in range(24) %}<div class="heatmap-cell heatmap-header">{{ h }}</div>{% endfor %}
                {% for day, name in [(1,'Mon'),(2,'Tue'),(3,'Wed'),(4,'Thu'),(5,'Fri'),(6,'Sat'),(7,'Sun')] %}
                <div class="heatmap-cell heatmap-day">{{ name }}</div>
                {% for h in range(24) %}
                {% set count = heatmap.get(day|string + '_' + h|string, 0) %}
                <div class="heatmap-cell heat-{{ [6, [0, count // 2]|max]|min }}" title="{{ count }}">{{ count if count else '' }}</div>
                {% endfor %}
                {% endfor %}
            </div>
        </div>
        
        <h2>Recent Detections</h2>
        <div class="recent-speeds">
            {% for s in recent_speeds %}
            <div class="speed-row {{ 'speeder' if s.speeder else '' }}">
                <span><span class="speed-value {{ 'over' if s.speeder else '' }}">{{ s.speed }}</span> / {{ s.limit }} mph</span>
                <span>{{ s.zone }} · {{ s.timestamp[11:19] }}</span>
            </div>
            {% endfor %}
            {% if not recent_speeds %}<div class="speed-row" style="justify-content:center;color:#888">No recent detections</div>{% endif %}
        </div>
    </div>
</body>
</html>
"""

CALIBRATE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Zone Calibration</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { text-align: center; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .calibration-area { display: flex; gap: 20px; flex-wrap: wrap; }
        .image-container { flex: 1; min-width: 600px; position: relative; }
        #calibration-image { width: 100%; border: 2px solid #333; border-radius: 8px; cursor: crosshair; }
        .sidebar { width: 300px; background: #2a2a2a; padding: 20px; border-radius: 8px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #aaa; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 4px; background: #1a1a1a; color: white; }
        .btn { background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; width: 100%; margin-bottom: 10px; }
        .btn:hover { background: #2563eb; }
        .btn:disabled { background: #4b5563; cursor: not-allowed; }
        .btn.secondary { background: #4b5563; }
        .btn.danger { background: #dc2626; }
        .point-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }
        .point-item.done { color: #4ade80; }
        .point-item.active { color: #3b82f6; }
        .existing-zone { display: flex; justify-content: space-between; align-items: center; padding: 10px; background: #3a3a3a; border-radius: 4px; margin-bottom: 8px; }
        .existing-zone .delete { background: #dc2626; border: none; color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
        svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Zone Calibration</h1>
        <a href="/" class="back-link">← Back to Live View</a>
        <div class="calibration-area">
            <div class="image-container">
                <img id="calibration-image" src="/calibration_frame" alt="Frame">
                <svg id="overlay"></svg>
            </div>
            <div class="sidebar">
                <div id="existing-zones"></div>
                <h3 style="margin-top:0">Add New Zone</h3>
                <div class="form-group"><label>Zone Name</label><input type="text" id="zone-name" value="Lane 1"></div>
                <div class="point-list">
                    <div class="point-item" id="point-0"><span>1. Top-Left</span><span id="coords-0">-</span></div>
                    <div class="point-item" id="point-1"><span>2. Top-Right</span><span id="coords-1">-</span></div>
                    <div class="point-item" id="point-2"><span>3. Bottom-Right</span><span id="coords-2">-</span></div>
                    <div class="point-item" id="point-3"><span>4. Bottom-Left</span><span id="coords-3">-</span></div>
                </div>
                <div class="form-group"><label>Width (ft)</label><input type="number" id="real-width" value="12" step="0.5"></div>
                <div class="form-group"><label>Height (ft)</label><input type="number" id="real-height" value="40" step="0.5"></div>
                <button class="btn" id="save-btn" disabled>Add Zone</button>
                <button class="btn secondary" id="reset-btn">Reset</button>
            </div>
        </div>
    </div>
    <script>
        const img = document.getElementById('calibration-image');
        const svg = document.getElementById('overlay');
        let points = [], existingZones = {{ zones | tojson }};
        const colors = ['#ff00ff', '#ffff00', '#ffa500', '#00ff00'];
        
        function render() {
            const el = document.getElementById('existing-zones');
            el.innerHTML = existingZones.length ? '<h3>Existing Zones</h3>' + existingZones.map((z,i) => 
                `<div class="existing-zone"><span style="color:${colors[i%4]}">${z.name}</span><button class="delete" onclick="deleteZone(${i})">×</button></div>`
            ).join('') : '<p style="color:#888">No zones</p>';
            
            const rect = img.getBoundingClientRect();
            const sx = rect.width / img.naturalWidth, sy = rect.height / img.naturalHeight;
            let html = '';
            existingZones.forEach((z, i) => {
                if (z.source_points?.length === 4) {
                    html += `<path d="${z.source_points.map((p,j) => (j?'L':'M')+p[0]*sx+','+p[1]*sy).join(' ')} Z" fill="${colors[i%4]}33" stroke="${colors[i%4]}" stroke-width="2"/>`;
                }
            });
            if (points.length >= 2) {
                html += `<path d="${points.map((p,i) => (i?'L':'M')+p[0]*sx+','+p[1]*sy).join(' ')}${points.length===4?' Z':''}" fill="rgba(59,130,246,0.2)" stroke="#3b82f6" stroke-width="2" stroke-dasharray="5,5"/>`;
            }
            svg.innerHTML = html;
        }
        
        window.deleteZone = function(i) {
            if (!confirm('Delete?')) return;
            fetch('/delete_zone', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({index:i}) })
                .then(r => r.json()).then(d => { existingZones = d.zones; render(); });
        };
        
        img.addEventListener('click', function(e) {
            if (points.length >= 4) return;
            const rect = img.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * img.naturalWidth / rect.width);
            const y = Math.round((e.clientY - rect.top) * img.naturalHeight / rect.height);
            points.push([x, y]);
            document.getElementById('coords-' + (points.length-1)).textContent = `${x}, ${y}`;
            document.getElementById('point-' + (points.length-1)).classList.add('done');
            if (points.length < 4) document.getElementById('point-' + points.length).classList.add('active');
            document.getElementById('save-btn').disabled = points.length < 4;
            render();
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            points = [];
            for (let i = 0; i < 4; i++) { document.getElementById('coords-'+i).textContent = '-'; document.getElementById('point-'+i).className = 'point-item'; }
            document.getElementById('point-0').classList.add('active');
            document.getElementById('save-btn').disabled = true;
            render();
        });
        
        document.getElementById('save-btn').addEventListener('click', () => {
            fetch('/add_zone', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({
                name: document.getElementById('zone-name').value,
                source_points: points,
                real_width: parseFloat(document.getElementById('real-width').value),
                real_height: parseFloat(document.getElementById('real-height').value)
            })}).then(r => r.json()).then(d => {
                existingZones = d.zones;
                points = [];
                for (let i = 0; i < 4; i++) { document.getElementById('coords-'+i).textContent = '-'; document.getElementById('point-'+i).className = 'point-item'; }
                document.getElementById('point-0').classList.add('active');
                document.getElementById('save-btn').disabled = true;
                document.getElementById('zone-name').value = 'Lane ' + (existingZones.length + 1);
                render();
            });
        });
        
        document.getElementById('point-0').classList.add('active');
        img.onload = render;
        render();
    </script>
</body>
</html>
"""

SCHEDULES_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speed Schedules</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { text-align: center; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .current-status { background: #2a2a2a; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; }
        .current-limit { font-size: 48px; font-weight: bold; color: {{ '#fbbf24' if 'School' in schedule_status else '#4ade80' }}; }
        .schedule-item { background: #2a2a2a; padding: 15px 20px; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .schedule-limit { font-size: 24px; font-weight: bold; color: #fbbf24; }
        .btn { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
        .btn.danger { background: #dc2626; }
        .add-form { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .form-row { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
        .form-group { flex: 1; min-width: 120px; }
        .form-group label { display: block; margin-bottom: 5px; color: #aaa; }
        .form-group input { width: 100%; padding: 10px; border: 1px solid #444; border-radius: 4px; background: #1a1a1a; color: white; }
        .days-select { display: flex; gap: 5px; flex-wrap: wrap; }
        .day-btn { padding: 8px 12px; border: 1px solid #444; border-radius: 4px; background: #1a1a1a; color: #888; cursor: pointer; }
        .day-btn.active { background: #3b82f6; border-color: #3b82f6; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speed Schedules</h1>
        <a href="/" class="back-link">← Back</a>
        <div class="current-status">
            <div>Current Limit</div>
            <div class="current-limit">{{ current_limit }} mph</div>
            <div style="color:#888">{{ schedule_status }}</div>
        </div>
        <h2>Schedules</h2>
        {% for s in schedules %}
        <div class="schedule-item">
            <div>
                <strong>{{ s.name }}</strong><br>
                <span style="color:#888;font-size:13px">{{ s.days | join(', ') | replace('1','Mon') | replace('2','Tue') | replace('3','Wed') | replace('4','Thu') | replace('5','Fri') | replace('6','Sat') | replace('7','Sun') }} · {{ s.start }}-{{ s.end }}</span>
            </div>
            <div style="display:flex;align-items:center;gap:15px">
                <span class="schedule-limit">{{ s.limit }} mph</span>
                <button class="btn danger" onclick="fetch('/delete_schedule',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:{{ loop.index0 }}})}).then(()=>location.reload())">×</button>
            </div>
        </div>
        {% else %}<p style="color:#888">No schedules</p>{% endfor %}
        
        <div class="add-form">
            <h3>Add Schedule</h3>
            <div class="form-row">
                <div class="form-group"><label>Name</label><input id="sched-name" value="School Hours"></div>
                <div class="form-group"><label>Limit (mph)</label><input type="number" id="sched-limit" value="20"></div>
            </div>
            <div class="form-row">
                <div class="form-group"><label>Start</label><input type="time" id="sched-start" value="07:30"></div>
                <div class="form-group"><label>End</label><input type="time" id="sched-end" value="08:30"></div>
            </div>
            <div class="form-group">
                <label>Days</label>
                <div class="days-select">
                    <button class="day-btn active" data-day="1">Mon</button>
                    <button class="day-btn active" data-day="2">Tue</button>
                    <button class="day-btn active" data-day="3">Wed</button>
                    <button class="day-btn active" data-day="4">Thu</button>
                    <button class="day-btn active" data-day="5">Fri</button>
                    <button class="day-btn" data-day="6">Sat</button>
                    <button class="day-btn" data-day="7">Sun</button>
                </div>
            </div>
            <button class="btn" onclick="addSchedule()">Add Schedule</button>
        </div>
    </div>
    <script>
        document.querySelectorAll('.day-btn').forEach(b => b.addEventListener('click', () => b.classList.toggle('active')));
        function addSchedule() {
            fetch('/add_schedule', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({
                name: document.getElementById('sched-name').value,
                limit: parseInt(document.getElementById('sched-limit').value),
                start: document.getElementById('sched-start').value,
                end: document.getElementById('sched-end').value,
                days: Array.from(document.querySelectorAll('.day-btn.active')).map(b => parseInt(b.dataset.day))
            })}).then(() => location.reload());
        }
    </script>
</body>
</html>
"""

SPEEDERS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Speeders</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; }
        .back-link { display: block; text-align: center; margin-bottom: 20px; color: #3b82f6; text-decoration: none; }
        .speeders-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }
        .speeder-card { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        .speeder-card img { width: 100%; height: 180px; object-fit: cover; }
        .speeder-info { padding: 15px; }
        .speeder-speed { font-size: 24px; font-weight: bold; color: #f87171; }
        .speeder-meta { color: #888; font-size: 13px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Captured Speeders</h1>
        <a href="/" class="back-link">← Back</a>
        {% if speeders %}
        <div class="speeders-grid">
            {% for s in speeders %}
            <div class="speeder-card">
                <img src="/speeder_image/{{ s.filename }}" alt="Speeder">
                <div class="speeder-info">
                    <div class="speeder-speed">{{ s.speed }} mph</div>
                    <div class="speeder-meta">{{ s.zone }} · {{ s.time }}</div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}<p style="text-align:center;color:#888">No speeders captured yet</p>{% endif %}
    </div>
</body>
</html>
"""

# =============================================================================
# VEHICLE TRACKER
# =============================================================================

class VehicleTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: {
            'positions': [], 'timestamps': [], 'speeds': [], 'last_speed': None,
            'zone_idx': None, 'captured': False, 'recorded': False,
            'crossed_stop_line': False, 'in_intersection': False,
            'violation_captured': False
        })
        self.transform_matrices = []
        self.zone_polygons = []
        self.update_transforms()
    
    def update_transforms(self):
        self.transform_matrices = []
        self.zone_polygons = []
        for zone in config.get("zones", []):
            if len(zone.get("source_points", [])) == 4:
                src = np.float32(zone["source_points"])
                dst = np.float32([[0, 0], [zone["real_width"], 0], 
                                  [zone["real_width"], zone["real_height"]], [0, zone["real_height"]]])
                self.transform_matrices.append(cv2.getPerspectiveTransform(src, dst))
                self.zone_polygons.append(src)
            else:
                self.transform_matrices.append(None)
                self.zone_polygons.append(None)
    
    def find_zone(self, point):
        for i, polygon in enumerate(self.zone_polygons):
            if polygon is not None and cv2.pointPolygonTest(polygon, point, False) >= 0:
                return i
        return None
    
    def pixel_to_feet(self, point, zone_idx):
        if zone_idx is None or zone_idx >= len(self.transform_matrices) or self.transform_matrices[zone_idx] is None:
            return None
        return cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), self.transform_matrices[zone_idx])[0][0]
    
    def update(self, track_id, bbox, timestamp):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = bbox[3]
        point = (center_x, center_y)
        
        track = self.tracks[track_id]
        track['positions'].append(point)
        track['timestamps'].append(timestamp)
        
        zone_idx = self.find_zone(point)
        if zone_idx is not None:
            track['zone_idx'] = zone_idx
        
        if len(track['positions']) > 30:
            track['positions'] = track['positions'][-30:]
            track['timestamps'] = track['timestamps'][-30:]
        
        if len(track['positions']) >= MIN_TRACK_LENGTH and track['zone_idx'] is not None:
            speed = self.calculate_speed(track_id)
            if speed is not None:
                track['speeds'].append(speed)
                if len(track['speeds']) > SPEED_SMOOTHING_WINDOW:
                    track['speeds'] = track['speeds'][-SPEED_SMOOTHING_WINDOW:]
                track['last_speed'] = np.mean(track['speeds'])
        
        # Check traffic light violations
        self.check_traffic_violations(track_id, point, bbox)
        
        return track['last_speed']
    
    def check_traffic_violations(self, track_id, point, bbox):
        """Check for stop line and red light violations"""
        global current_light_state
        
        track = self.tracks[track_id]
        tl_config = config.get("traffic_light")
        
        if not tl_config:
            return
        
        stop_line = tl_config.get("stop_line")
        intersection = tl_config.get("intersection_zone")
        
        # Check stop line crossing
        if stop_line and len(stop_line) == 2:
            front_of_car = (point[0], bbox[1])  # Top center of bounding box
            crossed = point_past_line(front_of_car, stop_line[0], stop_line[1], direction="down")
            
            if crossed and not track['crossed_stop_line']:
                track['crossed_stop_line'] = True
                
                # Stop line violation: crossed while red AND stopped (or very slow)
                speed = track.get('last_speed', 0) or 0
                if current_light_state == "red" and speed < 5:
                    if not track['violation_captured']:
                        track['violation_captured'] = True
                        return "stop_line"
        
        # Check intersection entry
        if intersection and len(intersection) >= 3:
            in_intersection = point_in_polygon(point, intersection)
            
            if in_intersection and not track['in_intersection']:
                track['in_intersection'] = True
                
                # Red light runner: entered intersection while red
                if current_light_state == "red":
                    if not track['violation_captured']:
                        track['violation_captured'] = True
                        return "red_light"
        
        return None
    
    def calculate_speed(self, track_id):
        track = self.tracks[track_id]
        zone_idx = track.get('zone_idx')
        if zone_idx is None or len(track['positions']) < 2:
            return None
        
        idx1, idx2 = 0, min(len(track['positions']) - 1, 4)
        pos1, pos2 = track['positions'][idx1], track['positions'][idx2]
        t1, t2 = track['timestamps'][idx1], track['timestamps'][idx2]
        
        real_pos1, real_pos2 = self.pixel_to_feet(pos1, zone_idx), self.pixel_to_feet(pos2, zone_idx)
        if real_pos1 is None or real_pos2 is None:
            return None
        
        distance = np.sqrt((real_pos2[0] - real_pos1[0])**2 + (real_pos2[1] - real_pos1[1])**2)
        time_diff = t2 - t1
        if time_diff <= 0:
            return None
        
        speed_mph = (distance / time_diff) * 0.681818
        return speed_mph if 0 <= speed_mph <= 150 else None
    
    def should_capture(self, track_id):
        track = self.tracks[track_id]
        if track['captured']:
            return False
        limit = get_current_speed_limit()
        if track['last_speed'] and track['last_speed'] > limit:
            track['captured'] = True
            return True
        return False
    
    def should_record(self, track_id):
        track = self.tracks[track_id]
        if track['recorded'] or track['last_speed'] is None:
            return False
        track['recorded'] = True
        return True
    
    def get_zone_name(self, track_id):
        track = self.tracks[track_id]
        zone_idx = track.get('zone_idx')
        if zone_idx is not None and zone_idx < len(config.get("zones", [])):
            return config["zones"][zone_idx].get("name", f"Zone {zone_idx + 1}")
        return "Unknown"
    
    def cleanup_old_tracks(self, current_time, max_age=2.0):
        to_remove = [tid for tid, t in self.tracks.items() 
                     if t['timestamps'] and (current_time - t['timestamps'][-1]) > max_age]
        for tid in to_remove:
            del self.tracks[tid]

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template_string(MAIN_TEMPLATE,
        zone_count=len(config.get("zones", [])),
        schedule_status=get_schedule_status(),
        current_limit=get_current_speed_limit(),
        light_state=current_light_state)

@app.route('/dashboard')
def dashboard():
    speeds = stats.get("speeds", [])
    speed_values = [s["speed"] for s in speeds if s.get("speed")]
    return render_template_string(DASHBOARD_TEMPLATE,
        total_vehicles=stats.get("total_vehicles", 0),
        total_speeders=stats.get("total_speeders", 0),
        speeder_percent=round(stats.get("total_speeders", 0) / max(stats.get("total_vehicles", 1), 1) * 100, 1),
        avg_speed=round(np.mean(speed_values), 1) if speed_values else 0,
        max_speed=round(max(speed_values), 1) if speed_values else 0,
        current_limit=get_current_speed_limit(),
        red_light_runners=stats.get("total_red_light_runners", 0),
        stop_line_violations=stats.get("total_stop_line_violations", 0),
        heatmap=stats.get("heatmap", {}),
        recent_speeds=list(reversed(speeds[-20:])))

@app.route('/traffic_light')
def traffic_light_page():
    return render_template_string(TRAFFIC_LIGHT_TEMPLATE,
        tl_config=config.get("traffic_light"),
        light_state=current_light_state)

@app.route('/save_traffic_light', methods=['POST'])
def save_traffic_light():
    data = request.json
    config["traffic_light"] = {
        "roi": data.get("roi", []),
        "stop_line": data.get("stop_line", []),
        "intersection_zone": data.get("intersection_zone", [])
    }
    save_config()
    return jsonify({"success": True})

@app.route('/clear_traffic_light', methods=['POST'])
def clear_traffic_light():
    config["traffic_light"] = None
    save_config()
    return jsonify({"success": True})

@app.route('/violations')
def violations_page():
    violation_list = []
    if os.path.exists(VIOLATIONS_DIR):
        for f in sorted(os.listdir(VIOLATIONS_DIR), reverse=True)[:50]:
            if f.endswith('.jpg'):
                parts = f.replace('.jpg', '').split('_')
                if len(parts) >= 3:
                    violation_list.append({
                        'filename': f,
                        'time': f"{parts[0]} {parts[1].replace('-', ':')}",
                        'type': parts[2],
                        'speed': parts[3].replace('mph', '') if len(parts) > 3 and 'mph' in parts[3] else None
                    })
    return render_template_string(VIOLATIONS_TEMPLATE,
        violations=violation_list,
        red_light_count=stats.get("total_red_light_runners", 0),
        stop_line_count=stats.get("total_stop_line_violations", 0))

@app.route('/violation_image/<filename>')
def violation_image(filename):
    filepath = os.path.join(VIOLATIONS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    return "Not found", 404

@app.route('/schedules')
def schedules():
    return render_template_string(SCHEDULES_TEMPLATE,
        schedules=config.get("speed_schedules", []),
        current_limit=get_current_speed_limit(),
        schedule_status=get_schedule_status())

@app.route('/add_schedule', methods=['POST'])
def add_schedule():
    data = request.json
    if "speed_schedules" not in config:
        config["speed_schedules"] = []
    config["speed_schedules"].append({
        "name": data.get("name", "Schedule"),
        "days": data.get("days", [1,2,3,4,5]),
        "start": data.get("start", "07:00"),
        "end": data.get("end", "08:00"),
        "limit": data.get("limit", 20)
    })
    save_config()
    return jsonify({"success": True})

@app.route('/delete_schedule', methods=['POST'])
def delete_schedule():
    index = request.json.get("index", -1)
    if 0 <= index < len(config.get("speed_schedules", [])):
        config["speed_schedules"].pop(index)
        save_config()
    return jsonify({"success": True})

@app.route('/calibrate')
def calibrate():
    return render_template_string(CALIBRATE_TEMPLATE, zones=config.get("zones", []))

@app.route('/add_zone', methods=['POST'])
def add_zone():
    global tracker
    data = request.json
    if "zones" not in config:
        config["zones"] = []
    config["zones"].append({
        "name": data["name"],
        "source_points": data["source_points"],
        "real_width": data["real_width"],
        "real_height": data["real_height"]
    })
    save_config()
    if tracker:
        tracker.update_transforms()
    return jsonify({"success": True, "zones": config["zones"]})

@app.route('/delete_zone', methods=['POST'])
def delete_zone():
    global tracker
    index = request.json.get("index", -1)
    if 0 <= index < len(config.get("zones", [])):
        config["zones"].pop(index)
        save_config()
        if tracker:
            tracker.update_transforms()
    return jsonify({"success": True, "zones": config.get("zones", [])})

@app.route('/speeders')
def speeders():
    speeder_list = []
    if os.path.exists(SPEEDERS_DIR):
        for f in sorted(os.listdir(SPEEDERS_DIR), reverse=True)[:50]:
            if f.endswith('.jpg'):
                parts = f.replace('.jpg', '').split('_')
                if len(parts) >= 3:
                    speeder_list.append({
                        'filename': f,
                        'time': f"{parts[0]} {parts[1].replace('-', ':')}",
                        'speed': parts[2].replace('mph', ''),
                        'zone': parts[3] if len(parts) > 3 else 'Unknown'
                    })
    return render_template_string(SPEEDERS_TEMPLATE, speeders=speeder_list)

@app.route('/speeder_image/<filename>')
def speeder_image(filename):
    filepath = os.path.join(SPEEDERS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    return "Not found", 404

@app.route('/calibration_frame')
def get_calibration_frame():
    global calibration_frame
    if calibration_frame is not None:
        ret, buffer = cv2.imencode('.jpg', calibration_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No frame available", 404

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

def start_web_server():
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def get_speed_color(speed, limit):
    if speed is None:
        return (128, 128, 128)
    elif speed <= limit:
        return (0, 255, 0)
    elif speed <= limit + 10:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def draw_overlays(frame):
    # Draw speed zones
    for i, zone in enumerate(config.get("zones", [])):
        if len(zone.get("source_points", [])) == 4:
            points = np.array(zone["source_points"], dtype=np.int32)
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            cv2.polylines(frame, [points], True, color, 2)
    
    # Draw traffic light config
    tl = config.get("traffic_light")
    if tl:
        # ROI
        if tl.get("roi") and len(tl["roi"]) == 4:
            x1, y1, x2, y2 = map(int, tl["roi"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Stop line
        if tl.get("stop_line") and len(tl["stop_line"]) == 2:
            p1, p2 = tl["stop_line"]
            cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 0, 255), 3)
        
        # Intersection zone
        if tl.get("intersection_zone") and len(tl["intersection_zone"]) >= 3:
            pts = np.array(tl["intersection_zone"], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    
    # Draw light state indicator
    light_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "unknown": (128, 128, 128)}
    cv2.circle(frame, (frame.shape[1] - 30, 30), 15, light_colors.get(current_light_state, (128, 128, 128)), -1)

def save_speeder_image(frame, bbox, speed, zone_name):
    x1, y1, x2, y2 = map(int, bbox)
    padding = 50
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{speed:.1f}mph_{zone_name.replace(' ', '-')}.jpg"
    cv2.imwrite(os.path.join(SPEEDERS_DIR, filename), crop)
    print(f"Saved speeder: {filename}")

def save_violation_image(frame, bbox, violation_type, speed=None):
    x1, y1, x2, y2 = map(int, bbox)
    padding = 80
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    speed_str = f"_{speed:.1f}mph" if speed else ""
    filename = f"{timestamp}_{violation_type}{speed_str}.jpg"
    cv2.imwrite(os.path.join(VIOLATIONS_DIR, filename), crop)
    print(f"Saved violation: {filename}")

# =============================================================================
# MAIN
# =============================================================================

tracker = None

def main(args):
    global output_frame, frame_lock, calibration_frame, tracker, current_light_state
    
    load_config()
    load_stats()
    
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    print(f"Connecting to: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)
    
    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream: {width}x{height} @ {fps} FPS")
    
    ret, calibration_frame = cap.read()
    
    print(f"Starting web server on http://0.0.0.0:{WEB_PORT}")
    Thread(target=start_web_server, daemon=True).start()
    
    tracker = VehicleTracker()
    frame_count = 0
    
    print(f"Starting detection with {len(config.get('zones', []))} zone(s)...")
    print(f"Traffic light configured: {config.get('traffic_light') is not None}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost connection, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(args.rtsp)
                continue
            
            if frame_count % 100 == 0:
                calibration_frame = frame.copy()
            
            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            current_time = time.time()
            current_limit = get_current_speed_limit()
            
            # Detect traffic light state
            detect_light_state(frame)
            
            if len(config.get("zones", [])) != len(tracker.transform_matrices):
                tracker.update_transforms()
            
            results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, 
                                  conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    speed = tracker.update(track_id, box, current_time)
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = get_speed_color(speed, current_limit)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if speed is not None:
                        zone_name = tracker.get_zone_name(track_id)
                        cv2.putText(frame, f"{speed:.1f}", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        is_speeder = speed > current_limit
                        if tracker.should_record(track_id):
                            record_speed(speed, zone_name, is_speeder)
                        
                        if tracker.should_capture(track_id):
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] SPEEDER: {speed:.1f} mph (limit: {current_limit})")
                            save_speeder_image(frame, box, speed, zone_name)
                    
                    # Check for traffic violations
                    track = tracker.tracks[track_id]
                    if track.get('violation_captured') and not track.get('violation_saved'):
                        track['violation_saved'] = True
                        if track.get('in_intersection') and current_light_state == "red":
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] RED LIGHT RUNNER!")
                            save_violation_image(frame, box, "red_light", speed)
                            record_violation("red_light", track_id, speed)
                        elif track.get('crossed_stop_line') and current_light_state == "red":
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] STOP LINE VIOLATION")
                            save_violation_image(frame, box, "stop_line", speed)
                            record_violation("stop_line", track_id, speed)
            
            draw_overlays(frame)
            
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Limit: {current_limit} mph | Light: {current_light_state.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            with frame_lock:
                output_frame = frame.copy()
            
            tracker.cleanup_old_tracks(current_time)
                    
    except KeyboardInterrupt:
        print("\nStopping...")
        save_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Speed & Traffic Detection")
    parser.add_argument('--rtsp', type=str, default=RTSP_URL, help='RTSP stream URL')
    args = parser.parse_args()
    main(args)
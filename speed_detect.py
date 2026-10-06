#!/usr/bin/env python3
"""
Vehicle Speed Detection with Traffic Light Monitoring
Features: Multi-zone speed detection, school zone scheduling, red light runner detection, stop line violations
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque, Counter
import time
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from threading import Thread, Lock, RLock
from flask import Flask, Response, render_template, request, jsonify, send_from_directory, abort
import json
import os
import shutil
import re
import logging
import sqlite3
import multiprocessing
import queue
from dotenv import load_dotenv
import psutil
import color_worker
from functools import wraps


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:PASSWORD@192.168.1.100:554/Streaming/Channels/2")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
DEFAULT_SPEED_LIMIT = int(os.getenv("DEFAULT_SPEED_LIMIT", "30"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))
PHOTO_RETENTION_HOURS = float(os.getenv("PHOTO_RETENTION_HOURS", "24"))  # 0 = keep forever
PHOTO_CLEANUP_INTERVAL_MINUTES = float(os.getenv("PHOTO_CLEANUP_INTERVAL_MINUTES", "15"))

CONFIG_FILE = "config.json"
STATS_FILE = "data/stats.json"
SPEEDERS_DIR = "data/speeders"
VIOLATIONS_DIR = "data/violations"
HARD_BRAKING_DIR = "data/hard_braking"
EVIDENCE_DIR = "data/evidence"        # never touched by photo cleanup
DB_FILE = "data/traffic.db"
EVIDENCE_SPEEDERS_PER_DAY = int(os.getenv("EVIDENCE_SPEEDERS_PER_DAY", "20"))
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MIN_TRACK_LENGTH = 5
SPEED_SMOOTHING_WINDOW = 3
VEHICLE_CLASSES = [2, 3, 5, 7]

ZONE_COLORS = [(255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0)]

CROP_SIZE_PRESETS = {
    "tight":  0.2,
    "medium": 0.5,
    "large":  1.0,
    "full":   None,
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

class InMemoryLogHandler(logging.Handler):
    """Custom logging handler that stores log records in a ring buffer."""

    def __init__(self, capacity=500):
        super().__init__()
        self.log_buffer = deque(maxlen=capacity)

    def emit(self, record):
        try:
            tz = ZoneInfo(config.get("timezone", "America/Chicago"))
        except (KeyError, Exception):
            tz = ZoneInfo("America/Chicago")
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=tz).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "message": self.format(record),
        }
        self.log_buffer.append(entry)

    def get_logs(self, level_filter=None, search=None, limit=500):
        logs = list(self.log_buffer)
        if level_filter and level_filter != "ALL":
            logs = [l for l in logs if l["level"] == level_filter]
        if search:
            search_lower = search.lower()
            logs = [l for l in logs if search_lower in l["message"].lower()]
        logs.reverse()
        return logs[:limit]

    def clear(self):
        self.log_buffer.clear()

memory_handler = InMemoryLogHandler(capacity=500)
memory_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger("speed_detection")
class RedactCredentialsFilter(logging.Filter):
    """Mask user:password in URLs (e.g. RTSP) before any handler sees the message."""
    pattern = re.compile(r'(\w+://)[^/@\s]+@')

    def filter(self, record):
        record.msg = self.pattern.sub(r'\1***@', record.getMessage())
        record.args = None
        return True

logger.setLevel(logging.DEBUG)
logger.addFilter(RedactCredentialsFilter())
logger.addHandler(memory_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# =============================================================================
# GLOBALS
# =============================================================================

app = Flask(__name__)
output_frame = None
frame_lock = Lock()
config_lock = RLock()   # protects config dict + tracker transforms + current_light_state
stats_lock = Lock()     # protects stats dict
calibration_frame = None
current_light_state = "unknown"  # red, yellow, green, unknown

config = {
    "zones": [],
    "speed_schedules": [
        {"name": "School Hours", "days": [1,2,3,4,5], "start": "07:30", "end": "08:30", "limit": 20},
        {"name": "School Hours", "days": [1,2,3,4,5], "start": "14:30", "end": "16:00", "limit": 20},
    ],
    "default_limit": DEFAULT_SPEED_LIMIT,
    "timezone": "America/Chicago",
    "traffic_light": None,  # {"enabled": False, "roi": [x1,y1,x2,y2], "stop_line": [[x1,y1],[x2,y2]], "intersection_zone": [[x1,y1],...]}
    "capture_stream_url": None,     # null = use detection stream
    "buffer_duration": 5.0,         # seconds of frames to retain
    "buffer_match_tolerance": 0.5,  # seconds — max time delta to match detections to buffer frames
    "snapshot_crop_size": "medium", # tight | medium | large
    "speed_smoothing_window": 3,    # rolling avg window
    "min_track_length": 5,          # positions before speed calc
    "process_every_n_frames": 2,    # frame skip
    "min_consecutive_over_limit": 1, # consecutive over-limit readings required
    "track_close_timeout": 2.0,     # seconds before track expires
    "speeder_zone_restriction": "anywhere",  # "anywhere" or "intersection_only"
    "tier_multipliers": {
        "full": 1.0,               # 3+ smoothed readings: speed > limit * 1.0
        "partial": 1.5,            # 1-2 readings: speed > limit * 1.5
    },
    "rapid_deceleration": {
        "enabled": False,
        "threshold": 15.0,              # mph/sec to trigger
        "min_initial_speed": 15.0,      # ignore vehicles already going slow
        "measurement_window": 4,        # frames per speed sample
        "save_images": True,
        "zone_restriction": "anywhere", # "anywhere" or "near_intersection"
        "near_intersection_feet": 100.0,
    },
    "speeder_thresholds": {
        "minor_pct": 0,     # 0% over limit = any amount over
        "major_pct": 25,    # 25% over limit (e.g., 25 mph in a 20 zone)
    },
}

def now_local():
    """Return current time in the configured timezone."""
    try:
        tz = ZoneInfo(config.get("timezone", "America/Chicago"))
    except (KeyError, Exception):
        tz = ZoneInfo("America/Chicago")
    return datetime.now(tz)

overlay_toggles = {
    "detections": True,
    "zones": True,
    "traffic_light": True,
    "light_indicator": True,
    "info_text": True,
}

stats = {
    "total_vehicles": 0,
    "total_speeders": 0,
    "total_red_light_runners": 0,
    "total_stop_line_violations": 0,
    "total_hard_braking": 0,
    "speeds": [],
    "heatmap": {},
    "violations": [],
    "hard_braking_events": []
}

# =============================================================================
# TRAFFIC LIGHT DETECTION
# =============================================================================

def detect_light_state(frame):
    """Detect traffic light state from ROI using color analysis.

    Uses a high brightness threshold to isolate the illuminated signal
    from the yellow housing/casing that would otherwise dominate detection.
    Applies margin requirement and temporal debouncing to prevent flickering
    when pixel counts are close (e.g. red housing reflecting yellow).
    """
    global current_light_state

    tl_config = config.get("traffic_light")
    if not tl_config or not tl_config.get("roi"):
        return "unknown"

    # Detection tuning — configurable via traffic light admin page
    det = tl_config.get("detection", {})
    debounce_frames = det.get("debounce_frames", 3)
    margin_pct = det.get("margin_pct", 25)
    bright_v = det.get("bright_v", 180)
    green_v = det.get("green_v", 120)
    red_min_pixels = det.get("red_min_pixels", 10)

    roi = tl_config["roi"]
    x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    # Extract ROI
    light_roi = frame[y1:y2, x1:x2]
    if light_roi.size == 0:
        return "unknown"

    # Reduce noise on small ROIs
    light_roi = cv2.GaussianBlur(light_roi, (3, 3), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)

    # Define color ranges (HSV) - require high brightness
    # Red has two ranges (wraps around 180 in hue)
    red_lower1 = np.array([0, 50, bright_v])
    red_upper1 = np.array([12, 255, 255])
    red_lower2 = np.array([155, 50, bright_v])
    red_upper2 = np.array([180, 255, 255])

    # Yellow signal (high saturation separates from white/washed-out pixels)
    yellow_lower = np.array([13, 80, bright_v])
    yellow_upper = np.array([35, 255, 255])

    # Green - lower V threshold (dimmer), wider hue range (can appear teal)
    green_lower = np.array([36, 30, green_v])
    green_upper = np.array([95, 255, 255])

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

    # Adaptive minimum - scale with ROI size, low floor for small lights
    roi_area = light_roi.shape[0] * light_roi.shape[1]
    min_pixels = max(3, int(roi_area * 0.005))

    # Determine raw winner from pixel counts
    if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > min_pixels:
        raw_state = "red"
        winner_pixels = red_pixels
    elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > min_pixels:
        raw_state = "yellow"
        winner_pixels = yellow_pixels
    elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > min_pixels:
        raw_state = "green"
        winner_pixels = green_pixels
    else:
        raw_state = "unknown"
        winner_pixels = 0

    # Red override: red doesn't appear naturally in this scene (no red
    # background behind the light), so any meaningful red pixel count means
    # the red signal is lit.  The yellow housing creates a persistent yellow
    # baseline that can mask the red-to-yellow comparison, but red pixels
    # alone are definitive.  Only apply when currently confirmed yellow
    # (the only state that physically transitions to red).
    if current_light_state == "yellow" and red_pixels >= red_min_pixels and raw_state != "red":
        raw_state = "red"
        winner_pixels = red_pixels

    prev_state = current_light_state

    # ---- Margin + debounce stabilization ----
    # Get pixel count for the currently confirmed state
    current_pixels = {"red": red_pixels, "yellow": yellow_pixels,
                      "green": green_pixels}.get(current_light_state, 0)

    # Margin check: new state must clearly beat current state
    margin_factor = 1.0 + margin_pct / 100.0
    passes_margin = (raw_state == current_light_state or
                     current_light_state == "unknown" or
                     winner_pixels > current_pixels * margin_factor)

    if raw_state == current_light_state:
        # Same state — reset candidate tracking
        detect_light_state._candidate = None
        detect_light_state._candidate_count = 0
    elif passes_margin:
        # Different state that passes margin — track as candidate
        candidate = getattr(detect_light_state, '_candidate', None)
        if raw_state == candidate:
            detect_light_state._candidate_count += 1
        else:
            detect_light_state._candidate = raw_state
            detect_light_state._candidate_count = 1

        # Commit state change only after enough consecutive frames agree
        if detect_light_state._candidate_count >= debounce_frames:
            current_light_state = raw_state
            detect_light_state._candidate = None
            detect_light_state._candidate_count = 0
    else:
        # Doesn't pass margin — reset candidate, keep current state
        detect_light_state._candidate = None
        detect_light_state._candidate_count = 0

    # Debug: log on every confirmed state change + periodic sampling
    if current_light_state != prev_state or getattr(detect_light_state, '_log_count', 0) % 50 == 0:
        v_channel = hsv[:, :, 2]
        max_v = int(v_channel.max()) if v_channel.size > 0 else 0
        avg_v = int(v_channel.mean()) if v_channel.size > 0 else 0
        candidate = getattr(detect_light_state, '_candidate', None)
        cand_count = getattr(detect_light_state, '_candidate_count', 0)
        logger.debug(f"[LIGHT] {prev_state}->{current_light_state} | "
                     f"raw:{raw_state} | "
                     f"R:{red_pixels} Y:{yellow_pixels} G:{green_pixels} | "
                     f"min_px:{min_pixels} roi:{roi_area}px | "
                     f"maxV:{max_v} avgV:{avg_v} | "
                     f"cand:{candidate}({cand_count}/{debounce_frames})")
    detect_light_state._log_count = getattr(detect_light_state, '_log_count', 0) + 1

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
    now = now_local()
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

def classify_speeder(speed, limit):
    if speed is None or speed <= limit:
        return None
    thresholds = config.get("speeder_thresholds", {})
    major_pct = thresholds.get("major_pct", 25)
    minor_pct = thresholds.get("minor_pct", 0)
    major_threshold = limit * (1 + major_pct / 100.0)
    if speed > major_threshold:
        return "major"
    minor_threshold = limit * (1 + minor_pct / 100.0)
    if speed > minor_threshold:
        return "minor"
    return None

# =============================================================================
# STATS FUNCTIONS
# =============================================================================

def load_stats():
    global stats
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            cutoff = (now_local() - timedelta(days=7)).isoformat()
            stats["speeds"] = [s for s in stats.get("speeds", []) if s.get("timestamp", "") > cutoff]
            stats["violations"] = [v for v in stats.get("violations", []) if v.get("timestamp", "") > cutoff]
            stats["hard_braking_events"] = [e for e in stats.get("hard_braking_events", []) if e.get("timestamp", "") > cutoff]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Could not load stats, starting fresh: {e}")

def save_stats():
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    tmp = STATS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(stats, f, cls=NumpyEncoder)
    os.replace(tmp, STATS_FILE)

def record_speed(speed, zone_name, is_speeder=False, tier=None):
    with stats_lock:
        now = now_local()
        stats["total_vehicles"] = stats.get("total_vehicles", 0) + 1

        if is_speeder:
            stats["total_speeders"] = stats.get("total_speeders", 0) + 1
            key = f"{now.isoweekday()}_{now.hour}"
            if "heatmap" not in stats:
                stats["heatmap"] = {}
            stats["heatmap"][key] = stats["heatmap"].get(key, 0) + 1

            if tier == "minor":
                stats["total_minor_speeders"] = stats.get("total_minor_speeders", 0) + 1
                if "heatmap_minor" not in stats:
                    stats["heatmap_minor"] = {}
                stats["heatmap_minor"][key] = stats["heatmap_minor"].get(key, 0) + 1
            elif tier == "major":
                stats["total_major_speeders"] = stats.get("total_major_speeders", 0) + 1
                if "heatmap_major" not in stats:
                    stats["heatmap_major"] = {}
                stats["heatmap_major"][key] = stats["heatmap_major"].get(key, 0) + 1

        if "speeds" not in stats:
            stats["speeds"] = []
        stats["speeds"].append({
            "timestamp": now.isoformat(),
            "speed": round(float(speed), 1),
            "zone": zone_name,
            "limit": int(get_current_speed_limit()),
            "speeder": bool(is_speeder),
            "tier": tier
        })
        if len(stats["speeds"]) > 1000:
            stats["speeds"] = stats["speeds"][-1000:]

        if stats["total_vehicles"] % 10 == 0:
            save_stats()

def record_violation(violation_type, track_id, speed=None):
    with stats_lock:
        now = now_local()

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

def record_hard_braking(track_id, decel_rate, initial_speed, final_speed, zone_name):
    with stats_lock:
        now = now_local()
        stats["total_hard_braking"] = stats.get("total_hard_braking", 0) + 1

        if "hard_braking_events" not in stats:
            stats["hard_braking_events"] = []
        stats["hard_braking_events"].append({
            "timestamp": now.isoformat(),
            "track_id": int(track_id),
            "decel_rate": round(float(decel_rate), 1),
            "initial_speed": round(float(initial_speed), 1),
            "final_speed": round(float(final_speed), 1),
            "zone": zone_name
        })
        if len(stats["hard_braking_events"]) > 500:
            stats["hard_braking_events"] = stats["hard_braking_events"][-500:]

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

            # Migration: remove old snapshot_stream_url, add new keys
            if "snapshot_stream_url" in config:
                old_snap = config.pop("snapshot_stream_url")
                if old_snap:
                    logger.info(f"Config migration: removed snapshot_stream_url ({old_snap})")

            # Ensure all new keys exist
            config.setdefault("capture_stream_url", None)
            config.setdefault("buffer_duration", 5.0)
            config.setdefault("speed_smoothing_window", 3)
            config.setdefault("min_track_length", 5)
            config.setdefault("process_every_n_frames", PROCESS_EVERY_N_FRAMES)
            config.setdefault("min_consecutive_over_limit", 1)
            config.setdefault("track_close_timeout", 2.0)
            config.setdefault("speeder_zone_restriction", "anywhere")
            config.setdefault("tier_multipliers", {"full": 1.0, "partial": 1.5})
            config.setdefault("timezone", "America/Chicago")

            logger.info(f"Config loaded from {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Could not load config: {e}")

def save_config():
    tmp = CONFIG_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)
    os.replace(tmp, CONFIG_FILE)
    logger.info(f"Config saved to {CONFIG_FILE}")

os.makedirs(SPEEDERS_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs(HARD_BRAKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

# =============================================================================
# PHOTO RETENTION
# =============================================================================

def cleanup_old_photos():
    """Delete captured photos older than PHOTO_RETENTION_HOURS. Returns (count, bytes)."""
    cutoff = time.time() - PHOTO_RETENTION_HOURS * 3600
    removed, freed = 0, 0
    for d in (SPEEDERS_DIR, VIOLATIONS_DIR, HARD_BRAKING_DIR):
        if not os.path.isdir(d):
            continue
        for entry in os.scandir(d):
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    size = entry.stat().st_size
                    os.remove(entry.path)
                    removed += 1
                    freed += size
            except FileNotFoundError:
                pass  # deleted concurrently (e.g. "clear" button in web UI)
            except OSError as e:
                logger.warning(f"Could not delete {entry.path}: {e}")
    return removed, freed

def photo_cleanup_loop():
    while True:
        try:
            removed, freed = cleanup_old_photos()
            if removed:
                logger.info(f"Photo cleanup: removed {removed} file(s) older than "
                            f"{PHOTO_RETENTION_HOURS:g}h, freed {freed / 1024**3:.2f} GB")
        except Exception as e:
            logger.error(f"Photo cleanup failed: {e}")
        time.sleep(PHOTO_CLEANUP_INTERVAL_MINUTES * 60)

# =============================================================================
# DATABASE
# =============================================================================
# Permanent per-vehicle history (stats.json only keeps running totals + last 1000).

db_conn = None
db_lock = Lock()

def init_db():
    global db_conn
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    db_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    db_conn.execute("PRAGMA journal_mode=WAL")
    db_conn.executescript("""
        CREATE TABLE IF NOT EXISTS vehicle_passes (
            id INTEGER PRIMARY KEY,
            ts TEXT NOT NULL,              -- local ISO time the vehicle left the scene
            lane TEXT,
            vehicle_class TEXT,
            color TEXT,
            color_confidence REAL,
            peak_speed REAL,
            speed_limit INTEGER,
            school_zone INTEGER,
            tier TEXT,                     -- minor / major / NULL
            reading_count INTEGER,
            in_intersection INTEGER,
            photo TEXT,                    -- file in data/speeders (deleted after retention period)
            evidence TEXT                  -- file in data/evidence (kept)
        );
        CREATE INDEX IF NOT EXISTS idx_passes_ts ON vehicle_passes(ts);
        CREATE TABLE IF NOT EXISTS stream_events (
            id INTEGER PRIMARY KEY,
            ts TEXT NOT NULL,
            event TEXT NOT NULL            -- connected / disconnected / reconnected
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY,
            ts TEXT NOT NULL,
            type TEXT NOT NULL,            -- red_light / stop_line / hard_braking
            speed REAL,
            details TEXT,                  -- JSON
            photo TEXT
        );
    """)
    db_conn.commit()
    logger.info(f"Database ready: {DB_FILE}")

def db_execute(sql, params=()):
    if db_conn is None:
        return None
    try:
        with db_lock:
            cur = db_conn.execute(sql, params)
            db_conn.commit()
            return cur
    except sqlite3.Error as e:
        logger.error(f"Database write failed: {e}")
        return None

def db_query(sql, params=()):
    if db_conn is None:
        return []
    with db_lock:
        return db_conn.execute(sql, params).fetchall()

def record_stream_event(event):
    db_execute("INSERT INTO stream_events (ts, event) VALUES (?, ?)",
               (now_local().isoformat(timespec="seconds"), event))

def record_event(event_type, speed=None, details=None, photo=None):
    db_execute("INSERT INTO events (ts, type, speed, details, photo) VALUES (?, ?, ?, ?, ?)",
               (now_local().isoformat(timespec="seconds"), event_type,
                round(float(speed), 1) if speed else None,
                json.dumps(details, cls=NumpyEncoder) if details else None, photo))

def keep_as_evidence(photo, speed, day):
    """Copy a speeder photo to EVIDENCE_DIR, keeping only the fastest N per day. Returns evidence filename or None."""
    if not photo or EVIDENCE_SPEEDERS_PER_DAY <= 0:
        return None
    kept = db_query("SELECT id, evidence, peak_speed FROM vehicle_passes "
                    "WHERE evidence IS NOT NULL AND ts LIKE ? ORDER BY peak_speed ASC", (day + "%",))
    if len(kept) >= EVIDENCE_SPEEDERS_PER_DAY:
        slowest_id, slowest_file, slowest_speed = kept[0]
        if speed <= slowest_speed:
            return None
        try:
            os.remove(os.path.join(EVIDENCE_DIR, slowest_file))
        except FileNotFoundError:
            pass
        db_execute("UPDATE vehicle_passes SET evidence = NULL WHERE id = ?", (slowest_id,))
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    try:
        shutil.copy2(os.path.join(SPEEDERS_DIR, photo), os.path.join(EVIDENCE_DIR, photo))
    except OSError as e:
        logger.warning(f"Could not keep evidence photo {photo}: {e}")
        return None
    return photo

# =============================================================================
# VEHICLE COLOR
# =============================================================================

# Colors are named by CLIP in a separate low-priority process (color_worker.py):
# torch thread settings are per-process, so running CLIP here would slow YOLO.

color_queue = None

def start_color_worker():
    global color_queue
    if os.getenv("VEHICLE_COLOR", "1") == "0":
        logger.info("Vehicle color disabled (VEHICLE_COLOR=0)")
        return
    ctx = multiprocessing.get_context("spawn")
    color_queue = ctx.Queue(maxsize=500)
    ctx.Process(target=color_worker.run, args=(color_queue, DB_FILE), daemon=True, name="color-worker").start()
    logger.info("Vehicle color worker started")

def is_grayscale_frame(frame):
    """True when the camera is in night/IR mode (whole picture has no color)."""
    thumb = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2HSV)
    return thumb[:, :, 1].mean() < 12

def queue_vehicle_color(row_id, frame, bbox):
    if color_queue is None or row_id is None or frame is None or bbox is None:
        return
    if is_grayscale_frame(frame):
        db_execute("UPDATE vehicle_passes SET color = 'unknown', color_confidence = 0 WHERE id = ?", (row_id,))
        return
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(fw, int(bbox[2])), min(fh, int(bbox[3]))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return
    ok, jpeg = cv2.imencode('.jpg', frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return
    try:
        color_queue.put_nowait((row_id, jpeg.tobytes()))
    except queue.Full:
        logger.debug(f"Color queue full, skipping vehicle {row_id}")

# =============================================================================
# VEHICLE TRACKER
# =============================================================================

class VehicleTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: {
            'positions': [], 'timestamps': [], 'speeds': [], 'last_speed': None,
            'zone_idx': None, 'captured': False, 'recorded': False,
            'crossed_stop_line': False, 'in_intersection': False,
            'violation_captured': False, 'violation_saved': False,
            'decel_captured': False,
            'speed_reading_count': 0,
            'peak_speed': 0.0,
            'consecutive_over_limit': 0,
            'max_consecutive_over_limit': 0,
            'best_capture': None,  # {'frame': np.array, 'bbox': tuple, 'score': float}
            'classes': Counter(),  # YOLO class id -> frames seen; majority wins (car/truck flicker)
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
        
        min_track_length = config.get("min_track_length", MIN_TRACK_LENGTH)
        smoothing_window = config.get("speed_smoothing_window", SPEED_SMOOTHING_WINDOW)

        if len(track['positions']) >= min_track_length and track['zone_idx'] is not None:
            speed = self.calculate_speed(track_id)
            if speed is not None:
                track['speeds'].append(speed)
                if len(track['speeds']) > smoothing_window:
                    track['speeds'] = track['speeds'][-smoothing_window:]
                track['last_speed'] = np.mean(track['speeds'])
                track['speed_reading_count'] += 1
                if track['last_speed'] > track['peak_speed']:
                    track['peak_speed'] = track['last_speed']
                limit = get_current_speed_limit()
                if track['last_speed'] > limit:
                    track['consecutive_over_limit'] += 1
                    if track['consecutive_over_limit'] > track['max_consecutive_over_limit']:
                        track['max_consecutive_over_limit'] = track['consecutive_over_limit']
                else:
                    track['consecutive_over_limit'] = 0
        
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
        # Crossing/intersection tracking always runs (speeder intersection_only uses it);
        # the flag only gates whether violations get flagged.
        violations_enabled = tl_config.get("enabled", False)
        
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
                if violations_enabled and current_light_state == "red" and speed < 5:
                    if not track['violation_captured']:
                        track['violation_captured'] = True

        # Check intersection entry
        if intersection and len(intersection) >= 3:
            in_intersection = point_in_polygon(point, intersection)

            if in_intersection and not track['in_intersection']:
                track['in_intersection'] = True

                # Red light runner: entered intersection while red
                if violations_enabled and current_light_state == "red":
                    if not track['violation_captured']:
                        track['violation_captured'] = True
    
    def calculate_speed(self, track_id):
        track = self.tracks[track_id]
        zone_idx = track.get('zone_idx')
        if zone_idx is None or len(track['positions']) < 2:
            return None
        
        idx2 = len(track['positions']) - 1
        idx1 = max(0, idx2 - 4)
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

    def _speed_between(self, track, idx1, idx2, zone_idx):
        """Calculate speed (mph) between two position buffer indices."""
        if idx1 < 0 or idx2 < 0 or idx1 >= len(track['positions']) or idx2 >= len(track['positions']):
            return None
        pos1, pos2 = track['positions'][idx1], track['positions'][idx2]
        t1, t2 = track['timestamps'][idx1], track['timestamps'][idx2]
        real_pos1 = self.pixel_to_feet(pos1, zone_idx)
        real_pos2 = self.pixel_to_feet(pos2, zone_idx)
        if real_pos1 is None or real_pos2 is None:
            return None
        distance = np.sqrt((real_pos2[0] - real_pos1[0])**2 + (real_pos2[1] - real_pos1[1])**2)
        time_diff = t2 - t1
        if time_diff <= 0:
            return None
        speed_mph = (distance / time_diff) * 0.681818
        return speed_mph if 0 <= speed_mph <= 150 else None

    def check_rapid_deceleration(self, track_id):
        """Check if vehicle is decelerating rapidly. Returns event dict or None."""
        track = self.tracks[track_id]
        if track.get('decel_captured'):
            return None

        decel_config = config.get("rapid_deceleration", {})
        if not decel_config.get("enabled"):
            return None

        zone_idx = track.get('zone_idx')
        if zone_idx is None:
            return None

        window = decel_config.get("measurement_window", 4)
        min_positions = (window * 2) + 1
        if len(track['positions']) < min_positions:
            return None

        # Two speed samples: earlier and recent
        idx_recent_end = len(track['positions']) - 1
        idx_recent_start = idx_recent_end - window
        idx_early_end = idx_recent_start
        idx_early_start = idx_early_end - window

        earlier_speed = self._speed_between(track, idx_early_start, idx_early_end, zone_idx)
        recent_speed = self._speed_between(track, idx_recent_start, idx_recent_end, zone_idx)
        if earlier_speed is None or recent_speed is None:
            return None

        # Check minimum initial speed
        min_speed = decel_config.get("min_initial_speed", 15.0)
        if earlier_speed < min_speed:
            return None

        # Zone restriction check
        restriction = decel_config.get("zone_restriction", "anywhere")
        if restriction == "near_intersection":
            tl_config = config.get("traffic_light")
            if tl_config and tl_config.get("stop_line") and len(tl_config["stop_line"]) == 2:
                stop_mid_px = ((tl_config["stop_line"][0][0] + tl_config["stop_line"][1][0]) / 2,
                               (tl_config["stop_line"][0][1] + tl_config["stop_line"][1][1]) / 2)
                current_px = track['positions'][-1]
                stop_mid_ft = self.pixel_to_feet(stop_mid_px, zone_idx)
                current_ft = self.pixel_to_feet(current_px, zone_idx)
                if stop_mid_ft is not None and current_ft is not None:
                    dist_ft = np.sqrt((current_ft[0] - stop_mid_ft[0])**2 + (current_ft[1] - stop_mid_ft[1])**2)
                    max_dist = decel_config.get("near_intersection_feet", 100.0)
                    if dist_ft > max_dist:
                        return None
                else:
                    return None
            else:
                return None

        # Compute deceleration rate
        time_span = track['timestamps'][idx_recent_end] - track['timestamps'][idx_early_start]
        if time_span <= 0:
            return None

        speed_drop = earlier_speed - recent_speed
        decel_rate = speed_drop / time_span  # mph/sec

        threshold = decel_config.get("threshold", 15.0)
        if decel_rate >= threshold:
            track['decel_captured'] = True
            return {
                "decel_rate": decel_rate,
                "initial_speed": earlier_speed,
                "final_speed": recent_speed
            }

        return None

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
    
    def _evaluate_speeder_tier(self, track, limit):
        """Evaluate whether a completed track qualifies as a speeder.

        Gate 1: minimum consecutive over-limit readings.
        Gate 2: tiered confidence thresholds based on reading count.
        Returns tier string ('minor'/'major') or None.
        """
        min_consec = config.get("min_consecutive_over_limit", 1)
        if track['max_consecutive_over_limit'] < min_consec:
            return None

        reading_count = track['speed_reading_count']
        peak = track['peak_speed']
        tier_mults = config.get("tier_multipliers", {"full": 1.0, "partial": 1.5})

        if reading_count >= 3:
            threshold = limit * tier_mults.get("full", 1.0)
        elif reading_count >= 1:
            threshold = limit * tier_mults.get("partial", 1.5)
        else:
            return None

        if peak <= threshold:
            return None

        return classify_speeder(peak, limit)

    def cleanup_old_tracks(self, current_time, max_age=2.0, frame_buffer=None):
        to_remove = [tid for tid, t in self.tracks.items()
                     if t['timestamps'] and (current_time - t['timestamps'][-1]) > max_age]
        for tid in to_remove:
            track = self.tracks[tid]
            if not track['recorded'] and track['last_speed'] is not None:
                peak_speed = float(track['peak_speed']) if track['peak_speed'] > 0 else float(track['last_speed'])
                zone_name = self.get_zone_name(tid)
                limit = get_current_speed_limit()
                tier = self._evaluate_speeder_tier(track, limit)

                # Zone restriction: skip capture if vehicle never entered intersection
                zone_restrict = config.get("speeder_zone_restriction", "anywhere")
                if zone_restrict == "intersection_only" and not track.get('in_intersection'):
                    tier = None

                photo = None
                best_capture = track.get('best_capture')
                if tier:
                    # Prefer proactive capture (grabbed while car was in frame)
                    if best_capture is not None:
                        logger.warning(f"SPEEDER ({tier.upper()}): {peak_speed:.1f} mph "
                                       f"(limit: {limit}, readings: {track['speed_reading_count']}) — source=proactive")
                        photo = save_speeder_image(best_capture['frame'], best_capture['bbox'], peak_speed, zone_name, tier)
                    elif frame_buffer is not None:
                        # Fallback to buffer search
                        best_frame, best_bbox, best_conf, best_ts = frame_buffer.find_best_frame(tid)
                        if best_frame is not None:
                            logger.warning(f"SPEEDER ({tier.upper()}): {peak_speed:.1f} mph "
                                           f"(limit: {limit}, readings: {track['speed_reading_count']}) — source=buffer")
                            photo = save_speeder_image(best_frame, best_bbox, peak_speed, zone_name, tier)
                        else:
                            logger.warning(f"SPEEDER ({tier.upper()}): {peak_speed:.1f} mph "
                                           f"(limit: {limit}, readings: {track['speed_reading_count']}) — no frame")

                record_speed(peak_speed, zone_name, tier is not None, tier)
                self._record_pass(track, tid, peak_speed, zone_name, limit, tier, photo, best_capture, frame_buffer)
            del self.tracks[tid]

    def _record_pass(self, track, tid, peak_speed, zone_name, limit, tier, photo, best_capture, frame_buffer):
        cls_id = track['classes'].most_common(1)[0][0] if track['classes'] else None
        ts = datetime.fromtimestamp(track['timestamps'][-1], tz=now_local().tzinfo).isoformat(timespec="seconds")
        default_limit = config.get("default_limit", DEFAULT_SPEED_LIMIT)
        evidence = keep_as_evidence(photo, peak_speed, ts[:10]) if tier == "major" else None
        cur = db_execute(
            "INSERT INTO vehicle_passes (ts, lane, vehicle_class, peak_speed, speed_limit, school_zone, "
            "tier, reading_count, in_intersection, photo, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, zone_name, VEHICLE_CLASS_NAMES.get(cls_id), round(peak_speed, 1),
             int(limit), int(limit < default_limit), tier, int(track['speed_reading_count']),
             int(bool(track.get('in_intersection'))), photo, evidence))

        # Color is filled in asynchronously by the color worker
        if best_capture is not None:
            col_frame, col_bbox = best_capture['frame'], best_capture['bbox']
        elif frame_buffer is not None:
            col_frame, col_bbox, _, _ = frame_buffer.find_best_frame(tid, copy=False)
        else:
            col_frame, col_bbox = None, None
        queue_vehicle_color(cur.lastrowid if cur else None, col_frame, col_bbox)

# =============================================================================
# FRAME BUFFER
# =============================================================================

class FrameBuffer:
    """In-memory ring buffer of detection frames for retrospective capture."""

    def __init__(self, duration=5.0, fps_estimate=15.0):
        self._lock = Lock()
        self._duration = duration
        self._fps = fps_estimate
        self._maxlen = max(1, int(duration * fps_estimate))
        self._buffer = deque(maxlen=self._maxlen)

    def add_frame(self, frame, timestamp, copy=True):
        """Store a frame + timestamp. copy=True when frame will be mutated later."""
        entry = {
            'frame': frame.copy() if copy else frame,
            'timestamp': timestamp,
            'detections': None,
        }
        with self._lock:
            self._buffer.append(entry)

    def attach_detections(self, timestamp, detections):
        """Walk backward from tail to find nearest matching timestamp, attach detections dict.

        detections: {track_id: {'bbox': (x1,y1,x2,y2), 'confidence': float}}
        Matches the closest buffer frame within buffer_match_tolerance seconds.
        """
        tolerance = config.get("buffer_match_tolerance", 0.5)
        with self._lock:
            best_entry = None
            best_delta = tolerance
            for entry in reversed(self._buffer):
                delta = abs(entry['timestamp'] - timestamp)
                if delta < best_delta:
                    best_delta = delta
                    best_entry = entry
                if timestamp - entry['timestamp'] > tolerance:
                    break
            if best_entry is not None:
                best_entry['detections'] = detections

    def find_best_frame(self, track_id, copy=True):
        """Score all entries with this track_id by bbox_area * confidence * edge_penalty.

        Returns (frame.copy(), bbox, confidence, timestamp) or (None, None, None, None).
        """
        best_score = -1
        best_result = (None, None, None, None)
        with self._lock:
            for entry in self._buffer:
                if entry['detections'] is None:
                    continue
                det = entry['detections'].get(track_id)
                if det is None:
                    continue
                bbox = det['bbox']
                conf = det['confidence']
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area = w * h
                # Edge penalty: 0.3 if bbox is clipped at frame edge
                frame_h, frame_w = entry['frame'].shape[:2]
                clipped = (bbox[0] <= 2 or bbox[1] <= 2 or
                           bbox[2] >= frame_w - 2 or bbox[3] >= frame_h - 2)
                edge_penalty = 0.3 if clipped else 1.0
                score = area * conf * edge_penalty
                if score > best_score:
                    best_score = score
                    best_result = (entry['frame'].copy() if copy else entry['frame'], bbox, conf, entry['timestamp'])
        return best_result

    def get_detection_frame(self, track_id, timestamp):
        """Buffer frame (capture resolution) matched to the detection at `timestamp`, plus its bbox.

        Returns the frame without copying — buffer frames are never modified after being added.
        """
        tolerance = config.get("buffer_match_tolerance", 0.5)
        with self._lock:
            for entry in reversed(self._buffer):
                if timestamp - entry['timestamp'] > tolerance:
                    break
                dets = entry['detections']
                if dets is not None and track_id in dets:
                    return entry['frame'], dets[track_id]['bbox']
        return None, None

    def find_recent_frame(self, track_id, max_age=1.0):
        """Walk backward, return most recent entry with this track_id.

        Returns (frame.copy(), bbox, confidence, timestamp) or (None, None, None, None).
        """
        with self._lock:
            now = self._buffer[-1]['timestamp'] if self._buffer else 0
            for entry in reversed(self._buffer):
                if now - entry['timestamp'] > max_age:
                    break
                if entry['detections'] is None:
                    continue
                det = entry['detections'].get(track_id)
                if det is not None:
                    return (entry['frame'].copy(), det['bbox'], det['confidence'], entry['timestamp'])
        return (None, None, None, None)

    def resize(self, duration, fps_estimate):
        """Rebuild buffer with new capacity. Called when user changes buffer_duration."""
        with self._lock:
            self._duration = duration
            self._fps = fps_estimate
            self._maxlen = max(1, int(duration * fps_estimate))
            old_entries = list(self._buffer)
            self._buffer = deque(maxlen=self._maxlen)
            for entry in old_entries[-self._maxlen:]:
                self._buffer.append(entry)

    @property
    def frame_count(self):
        with self._lock:
            return len(self._buffer)

    @property
    def duration(self):
        return self._duration

# =============================================================================
# CAPTURE STREAM READER (optional separate RTSP stream for high-res capture)
# =============================================================================

class CaptureStreamReader:
    """Background daemon thread that reads from a separate RTSP stream into the frame buffer."""

    def __init__(self, rtsp_url, frame_buffer, detection_resolution):
        self._url = rtsp_url
        self._frame_buffer = frame_buffer
        self._det_res = detection_resolution
        self._capture_res = None
        self._connected = False
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(f"Capture stream reader started: {self._url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def connected(self):
        return self._connected

    @property
    def scale_factors(self):
        """(sx, sy) for scaling detection bboxes to capture resolution."""
        if self._capture_res and self._det_res[0] > 0 and self._det_res[1] > 0:
            return (self._capture_res[0] / self._det_res[0],
                    self._capture_res[1] / self._det_res[1])
        return (1.0, 1.0)

    def _reader_loop(self):
        while self._running:
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                self._connected = False
                logger.warning(f"Capture stream failed to open: {self._url}, retrying in 2s")
                time.sleep(2)
                continue

            logger.info(f"Capture stream connected: {self._url}")
            self._connected = True

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Capture stream lost, reconnecting...")
                    self._connected = False
                    break
                if self._capture_res is None:
                    self._capture_res = (frame.shape[1], frame.shape[0])
                    logger.info(f"Capture stream resolution: {self._capture_res[0]}x{self._capture_res[1]}")
                self._frame_buffer.add_frame(frame, time.time(), copy=False)

            cap.release()
            if self._running:
                time.sleep(2)

# =============================================================================
# FLASK ROUTES
# =============================================================================

# =============================================================================
# FEATURE SWITCHES
# =============================================================================
# Disabled features are hidden from the UI entirely (nav, dashboard, routes 404).

def feature_enabled(name):
    if name == "traffic_violations":
        return bool((config.get("traffic_light") or {}).get("enabled", False))
    if name == "hard_braking":
        return bool(config.get("rapid_deceleration", {}).get("enabled", False))
    return False

def requires_feature(name):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not feature_enabled(name):
                abort(404)
            return view(*args, **kwargs)
        return wrapped
    return decorator

@app.context_processor
def inject_features():
    return {"features": {
        "traffic_violations": feature_enabled("traffic_violations"),
        "hard_braking": feature_enabled("hard_braking"),
    }}

@app.route('/save_features', methods=['POST'])
def save_features():
    data = request.json
    with config_lock:
        if "traffic_violations" in data:
            if config.get("traffic_light") is None:
                config["traffic_light"] = {"roi": [], "stop_line": [], "intersection_zone": [], "detection": {}}
            config["traffic_light"]["enabled"] = bool(data["traffic_violations"])
        if "hard_braking" in data:
            config.setdefault("rapid_deceleration", {})["enabled"] = bool(data["hard_braking"])
        save_config()
    logger.info(f"Features updated: {data}")
    return jsonify({"success": True})

@app.route('/')
def index():
    with config_lock:
        zone_count = len(config.get("zones", []))
        schedule_status = get_schedule_status()
        current_limit = get_current_speed_limit()
        light_state = current_light_state
    return render_template('index.html',
        zone_count=zone_count,
        schedule_status=schedule_status,
        current_limit=current_limit,
        light_state=light_state)

@app.route('/dashboard')
def dashboard():
    with config_lock:
        current_limit = get_current_speed_limit()
    with stats_lock:
        speeds = list(stats.get("speeds", []))
        total_vehicles = stats.get("total_vehicles", 0)
        total_speeders = stats.get("total_speeders", 0)
        total_minor_speeders = stats.get("total_minor_speeders", 0)
        total_major_speeders = stats.get("total_major_speeders", 0)
        red_light_runners = stats.get("total_red_light_runners", 0)
        stop_line_violations = stats.get("total_stop_line_violations", 0)
        total_hard_braking = stats.get("total_hard_braking", 0)
        heatmap = dict(stats.get("heatmap", {}))
        heatmap_minor = dict(stats.get("heatmap_minor", {}))
        heatmap_major = dict(stats.get("heatmap_major", {}))
    speed_values = [s["speed"] for s in speeds if s.get("speed")]
    return render_template('dashboard.html',
        total_vehicles=total_vehicles,
        total_speeders=total_speeders,
        total_minor_speeders=total_minor_speeders,
        total_major_speeders=total_major_speeders,
        speeder_percent=round(total_speeders / max(total_vehicles, 1) * 100, 1),
        avg_speed=round(np.mean(speed_values), 1) if speed_values else 0,
        max_speed=round(max(speed_values), 1) if speed_values else 0,
        current_limit=current_limit,
        red_light_runners=red_light_runners,
        stop_line_violations=stop_line_violations,
        total_hard_braking=total_hard_braking,
        heatmap=heatmap,
        heatmap_minor=heatmap_minor,
        heatmap_major=heatmap_major,
        recent_speeds=list(reversed(speeds[-20:])))

@app.route('/traffic_light')
@requires_feature("traffic_violations")
def traffic_light_page():
    with config_lock:
        tl_config = config.get("traffic_light")
        light_state = current_light_state
    with stats_lock:
        total_violations = stats.get("total_red_light_runners", 0) + stats.get("total_stop_line_violations", 0)
    return render_template('traffic_light.html',
        tl_config=tl_config,
        light_state=light_state,
        total_violations=total_violations)

@app.route('/save_traffic_light', methods=['POST'])
def save_traffic_light():
    data = request.json
    with config_lock:
        config["traffic_light"] = {
            "enabled": bool(data.get("enabled", False)),
            "roi": data.get("roi", []),
            "stop_line": data.get("stop_line", []),
            "intersection_zone": data.get("intersection_zone", []),
            "detection": data.get("detection", {})
        }
        save_config()
    return jsonify({"success": True})

@app.route('/clear_traffic_light', methods=['POST'])
def clear_traffic_light():
    with config_lock:
        config["traffic_light"] = None
        save_config()
    return jsonify({"success": True})

@app.route('/violations')
@requires_feature("traffic_violations")
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
    with stats_lock:
        red_light_count = stats.get("total_red_light_runners", 0)
        stop_line_count = stats.get("total_stop_line_violations", 0)
    return render_template('violations.html',
        violations=violation_list,
        red_light_count=red_light_count,
        stop_line_count=stop_line_count)

@app.route('/violation_image/<filename>')
@requires_feature("traffic_violations")
def violation_image(filename):
    return send_from_directory(VIOLATIONS_DIR, filename, mimetype='image/jpeg')

@app.route('/schedules')
def schedules():
    with config_lock:
        sched_list = list(config.get("speed_schedules", []))
        current_limit = get_current_speed_limit()
        schedule_status = get_schedule_status()
        default_limit = config.get("default_limit", DEFAULT_SPEED_LIMIT)
    return render_template('schedules.html',
        schedules=sched_list,
        current_limit=current_limit,
        schedule_status=schedule_status,
        default_limit=default_limit)

@app.route('/add_schedule', methods=['POST'])
def add_schedule():
    data = request.json
    with config_lock:
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
    with config_lock:
        if 0 <= index < len(config.get("speed_schedules", [])):
            config["speed_schedules"].pop(index)
            save_config()
    return jsonify({"success": True})

@app.route('/save_default_limit', methods=['POST'])
def save_default_limit():
    data = request.json
    limit = max(5, min(70, int(data.get("default_limit", 30))))
    with config_lock:
        config["default_limit"] = limit
        save_config()
    return jsonify({"success": True})

@app.route('/settings')
def settings():
    with config_lock:
        capture_url = config.get("capture_stream_url") or ""
        crop_size = config.get("snapshot_crop_size", "medium")
        buffer_duration = config.get("buffer_duration", 5.0)
        buffer_match_tolerance = config.get("buffer_match_tolerance", 0.5)
        thresholds = config.get("speeder_thresholds", {"minor_pct": 0, "major_pct": 25})
        current_limit = get_current_speed_limit()
        close_timeout = config.get("track_close_timeout", 2.0)
        speeder_zone_restriction = config.get("speeder_zone_restriction", "anywhere")
        tier_mults = config.get("tier_multipliers", {"full": 1.0, "partial": 1.5})
        smoothing_window = config.get("speed_smoothing_window", 3)
        min_track_len = config.get("min_track_length", 5)
        process_n_frames = config.get("process_every_n_frames", 2)
        min_consec_over = config.get("min_consecutive_over_limit", 1)
        current_timezone = config.get("timezone", "America/Chicago")
    capture_connected = capture_reader.connected if capture_reader else False
    buf_frames = frame_buffer.frame_count if frame_buffer else 0
    return render_template('settings.html',
                           capture_url=capture_url,
                           capture_connected=capture_connected,
                           buffer_duration=buffer_duration,
                           buffer_match_tolerance=buffer_match_tolerance,
                           buf_frames=buf_frames,
                           crop_size=crop_size,
                           presets=list(CROP_SIZE_PRESETS.keys()),
                           close_timeout=close_timeout,
                           speeder_zone_restriction=speeder_zone_restriction,
                           tier_full=tier_mults.get("full", 1.0),
                           tier_partial=tier_mults.get("partial", 1.5),
                           smoothing_window=smoothing_window,
                           min_track_len=min_track_len,
                           process_n_frames=process_n_frames,
                           min_consec_over=min_consec_over,
                           minor_pct=thresholds.get("minor_pct", 0),
                           major_pct=thresholds.get("major_pct", 25),
                           current_limit=current_limit,
                           current_timezone=current_timezone)

@app.route('/save_capture_settings', methods=['POST'])
def save_capture_settings():
    global capture_reader
    data = request.json
    new_url = (data.get("capture_stream_url") or "").strip() or None
    new_crop = data.get("crop_size", "medium")
    if new_crop not in CROP_SIZE_PRESETS:
        new_crop = "medium"
    new_duration = max(1.0, min(30.0, float(data.get("buffer_duration", 5.0))))
    new_tolerance = max(0.05, min(2.0, float(data.get("buffer_match_tolerance", 0.5))))

    with config_lock:
        old_url = config.get("capture_stream_url")
        old_duration = config.get("buffer_duration", 5.0)
        config["capture_stream_url"] = new_url
        config["snapshot_crop_size"] = new_crop
        config["buffer_duration"] = new_duration
        config["buffer_match_tolerance"] = new_tolerance
        save_config()

    # Resize buffer if duration changed
    if new_duration != old_duration and frame_buffer:
        fps_est = frame_buffer._fps
        frame_buffer.resize(new_duration, fps_est)
        logger.info(f"Frame buffer resized: {new_duration}s, {frame_buffer._maxlen} frames")

    # Restart capture reader if URL changed
    if new_url != old_url:
        if capture_reader:
            capture_reader.stop()
            capture_reader = None
        if new_url and frame_buffer:
            capture_reader = CaptureStreamReader(new_url, frame_buffer, detection_resolution)
            capture_reader.start()

    return jsonify({"success": True})

@app.route('/save_detection_tuning', methods=['POST'])
def save_detection_tuning():
    data = request.json
    with config_lock:
        config["speed_smoothing_window"] = max(1, min(10, int(data.get("speed_smoothing_window", 3))))
        config["min_track_length"] = max(2, min(15, int(data.get("min_track_length", 5))))
        config["process_every_n_frames"] = max(1, min(5, int(data.get("process_every_n_frames", 2))))
        config["min_consecutive_over_limit"] = max(1, min(5, int(data.get("min_consecutive_over_limit", 1))))
        save_config()
    return jsonify({"success": True})

@app.route('/save_capture_sensitivity', methods=['POST'])
def save_capture_sensitivity():
    data = request.json
    with config_lock:
        config["track_close_timeout"] = max(0.5, min(10.0, float(data.get("track_close_timeout", 2.0))))
        zone_val = data.get("speeder_zone_restriction", "anywhere")
        config["speeder_zone_restriction"] = zone_val if zone_val in ("anywhere", "intersection_only") else "anywhere"
        config["tier_multipliers"] = {
            "full": max(0.5, min(3.0, float(data.get("tier_full", 1.0)))),
            "partial": max(0.5, min(5.0, float(data.get("tier_partial", 1.5)))),
        }
        save_config()
    return jsonify({"success": True})

@app.route('/save_speeder_thresholds', methods=['POST'])
def save_speeder_thresholds():
    data = request.json
    with config_lock:
        config["speeder_thresholds"] = {
            "minor_pct": max(0, float(data.get("minor_pct", 0))),
            "major_pct": max(0, float(data.get("major_pct", 25))),
        }
        save_config()
    return jsonify({"success": True})

@app.route('/api/clear_speeders', methods=['POST'])
def api_clear_speeders():
    with stats_lock:
        if os.path.exists(SPEEDERS_DIR):
            shutil.rmtree(SPEEDERS_DIR)
            os.makedirs(SPEEDERS_DIR, exist_ok=True)
        stats['total_speeders'] = 0
        stats['total_minor_speeders'] = 0
        stats['total_major_speeders'] = 0
        stats['heatmap'] = {}
        stats['heatmap_minor'] = {}
        stats['heatmap_major'] = {}
        stats['speeds'] = []
        save_stats()
    logger.info("Speeder data cleared via web UI")
    return jsonify({"success": True})

@app.route('/api/clear_violations', methods=['POST'])
def api_clear_violations():
    with stats_lock:
        if os.path.exists(VIOLATIONS_DIR):
            shutil.rmtree(VIOLATIONS_DIR)
            os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        stats['total_red_light_runners'] = 0
        stats['total_stop_line_violations'] = 0
        stats['violations'] = []
        save_stats()
    logger.info("Violation data cleared via web UI")
    return jsonify({"success": True})

@app.route('/save_timezone', methods=['POST'])
def save_timezone():
    data = request.json
    tz_name = (data.get("timezone") or "America/Chicago").strip()
    try:
        ZoneInfo(tz_name)
    except (KeyError, Exception):
        return jsonify({"success": False, "error": f"Invalid timezone: {tz_name}"}), 400
    with config_lock:
        config["timezone"] = tz_name
        save_config()
    logger.info(f"Timezone changed to {tz_name}")
    return jsonify({"success": True})

@app.route('/calibrate')
def calibrate():
    with config_lock:
        zones = list(config.get("zones", []))
    return render_template('calibrate.html', zones=zones)

@app.route('/add_zone', methods=['POST'])
def add_zone():
    global tracker
    data = request.json
    with config_lock:
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
        zones = list(config["zones"])
    return jsonify({"success": True, "zones": zones})

@app.route('/edit_zone', methods=['POST'])
def edit_zone():
    global tracker
    data = request.json
    index = data.get("index", -1)
    with config_lock:
        zones = config.get("zones", [])
        if 0 <= index < len(zones):
            zones[index] = {
                "name": data["name"],
                "source_points": data["source_points"],
                "real_width": data["real_width"],
                "real_height": data["real_height"]
            }
            save_config()
            if tracker:
                tracker.update_transforms()
        zones = list(config.get("zones", []))
    return jsonify({"success": True, "zones": zones})

@app.route('/delete_zone', methods=['POST'])
def delete_zone():
    global tracker
    index = request.json.get("index", -1)
    with config_lock:
        if 0 <= index < len(config.get("zones", [])):
            config["zones"].pop(index)
            save_config()
            if tracker:
                tracker.update_transforms()
        zones = list(config.get("zones", []))
    return jsonify({"success": True, "zones": zones})

@app.route('/speeders')
def speeders():
    speeder_list = []
    if os.path.exists(SPEEDERS_DIR):
        for f in sorted(os.listdir(SPEEDERS_DIR), reverse=True)[:50]:
            if f.endswith('.jpg'):
                parts = f.replace('.jpg', '').split('_')
                if len(parts) >= 3:
                    tier = None
                    if len(parts) > 4 and parts[-1] in ('minor', 'major'):
                        tier = parts[-1]
                        zone = '_'.join(parts[3:-1])
                    else:
                        zone = parts[3] if len(parts) > 3 else 'Unknown'
                    speeder_list.append({
                        'filename': f,
                        'time': f"{parts[0]} {parts[1].replace('-', ':')}",
                        'speed': parts[2].replace('mph', ''),
                        'zone': zone,
                        'tier': tier
                    })
    return render_template('speeders.html', speeders=speeder_list)

@app.route('/speeder_image/<filename>')
def speeder_image(filename):
    return send_from_directory(SPEEDERS_DIR, filename, mimetype='image/jpeg')

@app.route('/hard_braking')
@requires_feature("hard_braking")
def hard_braking_page():
    event_list = []
    if os.path.exists(HARD_BRAKING_DIR):
        for f in sorted(os.listdir(HARD_BRAKING_DIR), reverse=True)[:50]:
            if f.endswith('.jpg'):
                parts = f.replace('.jpg', '').split('_')
                if len(parts) >= 4:
                    event_list.append({
                        'filename': f,
                        'time': f"{parts[0]} {parts[1].replace('-', ':')}",
                        'decel_rate': parts[2].replace('mphps', ''),
                        'initial_speed': parts[3].replace('mph', ''),
                        'zone': parts[4].replace('-', ' ') if len(parts) > 4 else 'Unknown'
                    })
    with stats_lock:
        total_hard_braking = stats.get("total_hard_braking", 0)
    return render_template('hard_braking.html',
        events=event_list,
        total_hard_braking=total_hard_braking)

@app.route('/hard_braking_image/<filename>')
@requires_feature("hard_braking")
def hard_braking_image(filename):
    return send_from_directory(HARD_BRAKING_DIR, filename, mimetype='image/jpeg')

@app.route('/deceleration_config')
@requires_feature("hard_braking")
def deceleration_config_page():
    with config_lock:
        decel_config = dict(config.get("rapid_deceleration", {}))
    with stats_lock:
        total_hard_braking = stats.get("total_hard_braking", 0)
    return render_template('rapid_deceleration.html',
        decel_config=decel_config,
        total_hard_braking=total_hard_braking)

@app.route('/save_rapid_deceleration', methods=['POST'])
def save_rapid_deceleration():
    data = request.json
    with config_lock:
        config["rapid_deceleration"] = {
            "enabled": bool(data.get("enabled", False)),
            "threshold": float(data.get("threshold", 15.0)),
            "min_initial_speed": float(data.get("min_initial_speed", 15.0)),
            "measurement_window": int(data.get("measurement_window", 4)),
            "save_images": bool(data.get("save_images", True)),
            "zone_restriction": data.get("zone_restriction", "anywhere"),
            "near_intersection_feet": float(data.get("near_intersection_feet", 100.0)),
        }
        save_config()
    return jsonify({"success": True})

@app.route('/logs')
def logs_page():
    return render_template('logs.html')

@app.route('/api/logs')
def api_logs():
    level_filter = request.args.get('level', 'ALL')
    search = request.args.get('search', '').strip()
    logs = memory_handler.get_logs(level_filter=level_filter, search=search)
    return jsonify({
        "logs": logs,
        "total": len(memory_handler.log_buffer)
    })

@app.route('/api/logs/clear', methods=['POST'])
def api_logs_clear():
    memory_handler.clear()
    logger.info("Log buffer cleared via web UI")
    return jsonify({"success": True})

@app.route('/api/system_stats')
def api_system_stats():
    proc = psutil.Process()
    with proc.oneshot():
        proc_cpu = proc.cpu_percent(interval=None)
        proc_mem = proc.memory_info().rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    return jsonify({
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(),
        "memory_used_mb": round(vm.used / (1024 * 1024)),
        "memory_total_mb": round(vm.total / (1024 * 1024)),
        "memory_percent": vm.percent,
        "process_cpu_percent": proc_cpu,
        "process_memory_mb": round(proc_mem, 1),
    })

@app.route('/api/overlays', methods=['GET'])
def api_overlays_get():
    return jsonify(overlay_toggles)

@app.route('/api/overlays', methods=['POST'])
def api_overlays_set():
    data = request.get_json()
    for key in overlay_toggles:
        if key in data:
            overlay_toggles[key] = bool(data[key])
    return jsonify(overlay_toggles)

@app.route('/api/light_state')
def api_light_state():
    det = {}
    tl_config = config.get("traffic_light")
    if tl_config:
        det = tl_config.get("detection", {})
    candidate = getattr(detect_light_state, '_candidate', None)
    cand_count = getattr(detect_light_state, '_candidate_count', 0)
    debounce = det.get("debounce_frames", 3)
    return jsonify({
        "state": current_light_state,
        "candidate": candidate,
        "candidate_count": cand_count,
        "debounce_frames": debounce
    })

@app.route('/calibration_frame')
def get_calibration_frame():
    with frame_lock:
        frame = calibration_frame.copy() if calibration_frame is not None else None
    if frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "No frame available", 404

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        with frame_lock:
            if output_frame is None:
                frame = None
            else:
                frame = output_frame.copy()
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

def start_web_server():
    app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)

# =============================================================================
# DRAWING HELPERS
# =============================================================================

def get_speed_color(speed, limit):
    if speed is None:
        return (128, 128, 128)
    thresholds = config.get("speeder_thresholds", {})
    major_pct = thresholds.get("major_pct", 25)
    minor_pct = thresholds.get("minor_pct", 0)
    major_threshold = limit * (1 + major_pct / 100.0)
    minor_threshold = limit * (1 + minor_pct / 100.0)
    if speed > major_threshold:
        return (0, 0, 255)       # red
    elif speed > minor_threshold:
        return (0, 255, 255)     # yellow
    else:
        return (0, 255, 0)       # green

def draw_overlays(frame):
    # Draw speed zones
    if overlay_toggles.get("zones", True):
        for i, zone in enumerate(config.get("zones", [])):
            if len(zone.get("source_points", [])) == 4:
                points = np.array(zone["source_points"], dtype=np.int32)
                color = ZONE_COLORS[i % len(ZONE_COLORS)]
                cv2.polylines(frame, [points], True, color, 2)

    # Draw traffic light config
    if overlay_toggles.get("traffic_light", True) and feature_enabled("traffic_violations"):
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
    if overlay_toggles.get("light_indicator", True) and feature_enabled("traffic_violations"):
        light_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255), "green": (0, 255, 0), "unknown": (128, 128, 128)}
        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, light_colors.get(current_light_state, (128, 128, 128)), -1)

def _crop_vehicle_direct(frame, bbox, crop_size_name):
    """Crop a vehicle from a buffer frame with configured padding.

    Buffer frames are at detection resolution — no scaling needed.
    Returns (crop, vehicle_rect) where vehicle_rect is the (x1, y1, x2, y2)
    bounding box in crop-local coordinates.
    """
    bx1, by1, bx2, by2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    factor = CROP_SIZE_PRESETS.get(crop_size_name, CROP_SIZE_PRESETS["medium"])
    if factor is None:
        # Full frame, no crop
        vr = (int(bx1), int(by1), int(bx2), int(by2))
        return frame.copy(), vr

    pad = max(bx2 - bx1, by2 - by1) * factor
    h, w = frame.shape[:2]
    x1 = int(max(0, bx1 - pad))
    y1 = int(max(0, by1 - pad))
    x2 = int(min(w, bx2 + pad))
    y2 = int(min(h, by2 + pad))
    crop = frame[y1:y2, x1:x2].copy()  # copy: callers draw on it, and buffer frames are shared
    if crop.size == 0:
        return frame.copy(), (0, 0, frame.shape[1], frame.shape[0])
    # Vehicle bbox relative to crop origin
    vr = (int(bx1) - x1, int(by1) - y1, int(bx2) - x1, int(by2) - y1)
    return crop, vr


def save_speeder_image(frame, bbox, speed, zone_name, tier="minor"):
    crop_size = config.get("snapshot_crop_size", "medium")
    crop, vr = _crop_vehicle_direct(frame, bbox, crop_size)
    # Always annotate — buffer frames have no overlays
    current_limit = get_current_speed_limit()
    color = get_speed_color(speed, current_limit)
    cv2.rectangle(crop, (vr[0], vr[1]), (vr[2], vr[3]), color, 2)
    cv2.putText(crop, f"{speed:.1f} mph", (vr[0], vr[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    timestamp = now_local().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{speed:.1f}mph_{zone_name.replace(' ', '-')}_{tier}.jpg"
    os.makedirs(SPEEDERS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(SPEEDERS_DIR, filename), crop)
    logger.info(f"Saved speeder ({tier}): {filename} (source=buffer)")
    return filename

def save_violation_image(frame, bbox, violation_type, speed=None):
    crop_size = config.get("snapshot_crop_size", "medium")
    crop, vr = _crop_vehicle_direct(frame, bbox, crop_size)
    cv2.rectangle(crop, (vr[0], vr[1]), (vr[2], vr[3]), (0, 0, 255), 2)
    label = violation_type.replace("_", " ").upper()
    if speed:
        label += f" {speed:.1f} mph"
    cv2.putText(crop, label, (vr[0], vr[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    timestamp = now_local().strftime("%Y-%m-%d_%H-%M-%S")
    speed_str = f"_{speed:.1f}mph" if speed else ""
    filename = f"{timestamp}_{violation_type}{speed_str}.jpg"
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(VIOLATIONS_DIR, filename), crop)
    logger.info(f"Saved violation: {filename} (source=buffer)")
    return filename

def save_hard_braking_image(frame, bbox, decel_rate, initial_speed, zone_name):
    crop_size = config.get("snapshot_crop_size", "medium")
    crop, vr = _crop_vehicle_direct(frame, bbox, crop_size)
    cv2.rectangle(crop, (vr[0], vr[1]), (vr[2], vr[3]), (0, 165, 255), 2)
    label = f"BRAKING {decel_rate:.0f} mph/s @ {initial_speed:.0f} mph"
    cv2.putText(crop, label, (vr[0], vr[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    timestamp = now_local().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{decel_rate:.0f}mphps_{initial_speed:.0f}mph_{zone_name.replace(' ', '-')}.jpg"
    os.makedirs(HARD_BRAKING_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(HARD_BRAKING_DIR, filename), crop)
    logger.info(f"Saved hard braking: {filename} (source=buffer)")
    return filename

frame_buffer = None
capture_reader = None
detection_resolution = (0, 0)

# =============================================================================
# MAIN
# =============================================================================

tracker = None

def main(args):
    global output_frame, frame_lock, calibration_frame, tracker, current_light_state
    global detection_resolution, frame_buffer, capture_reader

    load_config()
    load_stats()
    init_db()
    start_color_worker()

    logger.info("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    logger.info(f"Connecting to: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)

    if not cap.isOpened():
        logger.error("Could not open RTSP stream")
        return

    record_stream_event("connected")
    stream_down = False

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detection_resolution = (width, height)
    logger.info(f"Stream: {width}x{height} @ {fps} FPS")

    # Set up frame buffer for retrospective capture
    buffer_duration = config.get("buffer_duration", 5.0)
    frame_buffer = FrameBuffer(duration=buffer_duration, fps_estimate=fps or 15.0)
    logger.info(f"Frame buffer initialized: {frame_buffer._maxlen} frames, {buffer_duration}s")

    capture_reader = None
    capture_url = config.get("capture_stream_url")
    if capture_url:
        capture_reader = CaptureStreamReader(capture_url, frame_buffer, detection_resolution)
        capture_reader.start()
    else:
        logger.info("Using detection stream for capture buffer")

    ret, calibration_frame = cap.read()

    logger.info(f"Starting web server on http://0.0.0.0:{WEB_PORT}")
    Thread(target=start_web_server, daemon=True).start()

    if PHOTO_RETENTION_HOURS > 0:
        logger.info(f"Photo retention: {PHOTO_RETENTION_HOURS:g}h, checking every {PHOTO_CLEANUP_INTERVAL_MINUTES:g} min")
        Thread(target=photo_cleanup_loop, daemon=True).start()
    
    tracker = VehicleTracker()
    frame_count = 0
    
    logger.info(f"Starting detection with {len(config.get('zones', []))} zone(s)...")
    logger.info(f"Traffic light configured: {config.get('traffic_light') is not None}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if not stream_down:
                    stream_down = True
                    record_stream_event("disconnected")
                logger.warning("Lost connection, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(args.rtsp)
                if not cap.isOpened():
                    logger.error("Reconnect failed, retrying in 10s...")
                    time.sleep(10)
                continue
            if stream_down:
                stream_down = False
                record_stream_event("reconnected")
            
            frame_time = time.time()
            if capture_reader is None:  # same-stream mode
                frame_buffer.add_frame(frame, frame_time, copy=True)

            if frame_count % 100 == 0:
                with frame_lock:
                    calibration_frame = frame.copy()

            frame_count += 1
            process_n = config.get("process_every_n_frames", PROCESS_EVERY_N_FRAMES)
            if frame_count % process_n != 0:
                continue

            current_time = frame_time

            with config_lock:
                current_limit = get_current_speed_limit()
                if feature_enabled("traffic_violations"):
                    detect_light_state(frame)
                if len(config.get("zones", [])) != len(tracker.transform_matrices):
                    tracker.update_transforms()

            results = model.track(frame, persist=True, classes=VEHICLE_CLASSES,
                                  conf=CONFIDENCE_THRESHOLD, verbose=False)

            with config_lock:
                # Re-check in case zones changed during inference
                if len(config.get("zones", [])) != len(tracker.transform_matrices):
                    tracker.update_transforms()
                light_state = current_light_state

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    # Build detections dict and attach to buffer
                    detections_for_buffer = {}
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        detections_for_buffer[int(track_id)] = {
                            'bbox': tuple(float(v) for v in box),
                            'confidence': float(conf),
                        }

                    if capture_reader is not None:
                        sx, sy = capture_reader.scale_factors
                        scaled_dets = {}
                        for tid, det in detections_for_buffer.items():
                            b = det['bbox']
                            scaled_dets[tid] = {
                                'bbox': (b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy),
                                'confidence': det['confidence'],
                            }
                        frame_buffer.attach_detections(frame_time, scaled_dets)
                    else:
                        frame_buffer.attach_detections(frame_time, detections_for_buffer)

                    clean_frame_ref = [None]  # lazy copy, shared across vehicles in this frame

                    for box, track_id, det_conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                        speed = tracker.update(track_id, box, current_time)
                        tracker.tracks[track_id]['classes'][int(cls_id)] += 1

                        # Proactive capture: score this detection while car is in frame
                        if speed is not None and speed > current_limit:
                            track_cap = tracker.tracks[track_id]
                            w = float(box[2] - box[0])
                            h = float(box[3] - box[1])
                            area = w * h
                            fh, fw = frame.shape[:2]
                            clipped = (box[0] <= 2 or box[1] <= 2 or
                                       box[2] >= fw - 2 or box[3] >= fh - 2)
                            edge_penalty = 0.3 if clipped else 1.0
                            cap_score = area * float(det_conf) * edge_penalty
                            best = track_cap.get('best_capture')
                            if best is None or cap_score > best['score']:
                                # Prefer the full-resolution capture-stream frame matched to this detection
                                hires_frame, hires_bbox = frame_buffer.get_detection_frame(int(track_id), frame_time)
                                if hires_frame is not None:
                                    track_cap['best_capture'] = {
                                        'frame': hires_frame,
                                        'bbox': hires_bbox,
                                        'score': cap_score,
                                    }
                                else:
                                    if clean_frame_ref[0] is None:
                                        clean_frame_ref[0] = frame.copy()
                                    track_cap['best_capture'] = {
                                        'frame': clean_frame_ref[0],
                                        'bbox': tuple(float(v) for v in box),
                                        'score': cap_score,
                                    }

                        if overlay_toggles.get("detections", True):
                            x1, y1, x2, y2 = map(int, box)
                            color = get_speed_color(speed, current_limit)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            if speed is not None:
                                cv2.putText(frame, f"{speed:.1f}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            # Lane/zone label on right side of bounding box
                            zone_name = tracker.get_zone_name(track_id)
                            if zone_name != "Unknown":
                                cv2.putText(frame, zone_name, (x2 + 4, y1 + 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                        # Check for traffic violations
                        track = tracker.tracks[track_id]
                        if track.get('violation_captured') and not track.get('violation_saved'):
                            track['violation_saved'] = True
                            buf_frame, buf_bbox, _, _ = frame_buffer.find_recent_frame(track_id)
                            if buf_frame is None:
                                buf_frame, buf_bbox = frame.copy(), tuple(float(v) for v in box)
                            if track.get('in_intersection') and light_state == "red":
                                logger.warning("RED LIGHT RUNNER!")
                                vphoto = save_violation_image(buf_frame, buf_bbox, "red_light", speed)
                                record_violation("red_light", track_id, speed)
                                record_event("red_light", speed, photo=vphoto)
                            elif track.get('crossed_stop_line') and light_state == "red":
                                logger.warning("STOP LINE VIOLATION")
                                vphoto = save_violation_image(buf_frame, buf_bbox, "stop_line", speed)
                                record_violation("stop_line", track_id, speed)
                                record_event("stop_line", speed, photo=vphoto)

                        # Check for rapid deceleration
                        if not track.get('decel_captured'):
                            decel_event = tracker.check_rapid_deceleration(track_id)
                            if decel_event:
                                zone_name = tracker.get_zone_name(track_id)
                                logger.warning(f"HARD BRAKING: {decel_event['decel_rate']:.1f} mph/s "
                                               f"({decel_event['initial_speed']:.1f} -> {decel_event['final_speed']:.1f} mph)")
                                decel_cfg = config.get("rapid_deceleration", {})
                                hb_photo = None
                                if decel_cfg.get("save_images", True):
                                    hb_frame, hb_bbox, _, _ = frame_buffer.find_recent_frame(track_id)
                                    if hb_frame is None:
                                        hb_frame, hb_bbox = frame.copy(), tuple(float(v) for v in box)
                                    hb_photo = save_hard_braking_image(hb_frame, hb_bbox, decel_event['decel_rate'],
                                                                      decel_event['initial_speed'], zone_name)
                                record_hard_braking(track_id, decel_event['decel_rate'],
                                                   decel_event['initial_speed'], decel_event['final_speed'], zone_name)
                                record_event("hard_braking", decel_event['initial_speed'],
                                             {"decel_rate": decel_event['decel_rate'],
                                              "final_speed": decel_event['final_speed'], "lane": zone_name},
                                             photo=hb_photo)

                        # Draw braking overlay
                        if track.get('decel_captured') and overlay_toggles.get("detections", True):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.putText(frame, "BRAKING", (x1, y2 + 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                draw_overlays(frame)

            if overlay_toggles.get("info_text", True):
                h, w = frame.shape[:2]
                line1 = now_local().strftime("%Y-%m-%d %H:%M:%S")
                line2 = f"Limit: {current_limit} mph | Light: {light_state.upper()}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw1, th1), _ = cv2.getTextSize(line1, font, 0.7, 2)
                (tw2, th2), _ = cv2.getTextSize(line2, font, 0.6, 2)
                pad = 8
                box_w = max(tw1, tw2) + pad * 2
                box_h = th1 + th2 + pad * 3
                cv2.rectangle(frame, (0, h - box_h), (box_w, h), (0, 0, 0), -1)
                cv2.putText(frame, line1, (pad, h - box_h + pad + th1),
                           font, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, line2, (pad, h - pad),
                           font, 0.6, (0, 255, 255), 2)

            with frame_lock:
                output_frame = frame.copy()

            close_timeout = config.get("track_close_timeout", 2.0)
            tracker.cleanup_old_tracks(current_time, max_age=close_timeout, frame_buffer=frame_buffer)
                    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        save_stats()
        if capture_reader:
            capture_reader.stop()
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Speed & Traffic Detection")
    parser.add_argument('--rtsp', type=str, default=RTSP_URL, help='RTSP stream URL')
    args = parser.parse_args()
    main(args)
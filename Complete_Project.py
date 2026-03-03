import cv2
import json
import numpy as np
import requests
import os
import uuid
import sqlite3
import secrets
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, url_for, redirect, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from functools import wraps
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import qrcode
from io import BytesIO
import base64
load_dotenv()

# MediaPipe Tasks API (pose_landmarker)
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarksConnections
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_lib
# ==============================================================================
# 1. FLASK APPLICATION SETUP
# ==============================================================================

app = Flask(__name__, static_folder='static')

# --- Configuration ---
# Replace with your actual key
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set. Check your .env file.")
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apex_cric.db')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
POSE_MODEL_PATH = os.path.join(MODEL_DIR, 'pose_landmarker_heavy.task')


# ==============================================================================
# AUTH: SQLite user store and helpers
# ==============================================================================

def get_db():
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create users and analyses tables if they do not exist."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_filename TEXT,
                output_full_filename TEXT NOT NULL,
                output_skeleton_filename TEXT NOT NULL,
                release_features TEXT,
                ai_feedback TEXT,
                video_width INTEGER,
                video_height INTEGER,
                fps REAL
            )
        """)
        try:
            conn.execute("ALTER TABLE analyses ADD COLUMN frame_data TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                age INTEGER,
                height REAL,
                weight REAL,
                bowling_arm TEXT,
                bowling_style TEXT,
                team_name TEXT,
                photo_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            conn.execute("ALTER TABLE analyses ADD COLUMN player_id INTEGER REFERENCES players(id)")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                share_id TEXT UNIQUE NOT NULL,
                plan_type TEXT,
                plan_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                user_id INTEGER NOT NULL PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                token TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def get_user_by_email(email):
    """Return user row (dict-like) or None."""
    with get_db() as conn:
        row = conn.execute("SELECT id, email, password_hash, name FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Return user row or None."""
    with get_db() as conn:
        row = conn.execute("SELECT id, email, name FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_auth_token(user_id):
    """Create or replace a long-lived auth token for the user. Returns the token string."""
    token = secrets.token_urlsafe(32)
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO auth_tokens (user_id, token) VALUES (?, ?)",
            (user_id, token),
        )
        conn.commit()
    return token


def lookup_user_by_token(token):
    """Return user_id for a valid token, else None."""
    if not token or not isinstance(token, str):
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT user_id FROM auth_tokens WHERE token = ?",
            (token.strip(),),
        ).fetchone()
    return row["user_id"] if row else None


def create_user(email, password, name=None):
    """Create user. Returns (user_dict, None) or (None, error_message)."""
    email = email.strip().lower()
    if not email or not password:
        return None, "Email and password are required."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."
    if get_user_by_email(email):
        return None, "An account with this email already exists."
    password_hash = generate_password_hash(password, method="scrypt")
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, name) VALUES (?, ?, ?)",
            (email, password_hash, (name or "").strip() or None),
        )
        conn.commit()
        user_id = cur.lastrowid
    return get_user_by_id(user_id), None


def get_user_by_id_with_password(user_id):
    """Return user row including password_hash, or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, name FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def update_user(user_id, name=None, email=None, new_password=None, current_password=None):
    """Update user profile. Returns (user_dict, None) or (None, error_message)."""
    user = get_user_by_id_with_password(user_id)
    if not user:
        return None, "User not found."
    if new_password:
        if not current_password:
            return None, "Current password is required to set a new password."
        if not check_password_hash(user["password_hash"], current_password):
            return None, "Current password is incorrect."
        if len(new_password) < 6:
            return None, "New password must be at least 6 characters."
    updates = []
    params = []
    if name is not None:
        updates.append("name = ?")
        params.append((name or "").strip() or None)
    if email is not None:
        email = email.strip().lower()
        if not email:
            return None, "Email cannot be empty."
        other = get_user_by_email(email)
        if other and other["id"] != user_id:
            return None, "An account with this email already exists."
        updates.append("email = ?")
        params.append(email)
    if new_password:
        updates.append("password_hash = ?")
        params.append(generate_password_hash(new_password, method="scrypt"))
    if not updates:
        return get_user_by_id(user_id), None
    params.append(user_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()
    return get_user_by_id(user_id), None


def save_analysis(user_id, original_filename, output_full_filename, output_skeleton_filename,
                  release_features, ai_feedback, video_width, video_height, fps, frame_data=None, player_id=None):
    """Save an analysis record for the user. Returns the new analysis id."""
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO analyses (user_id, player_id, original_filename, output_full_filename, output_skeleton_filename,
               release_features, ai_feedback, video_width, video_height, fps, frame_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                player_id or None,
                original_filename or None,
                output_full_filename,
                output_skeleton_filename,
                json.dumps(release_features) if release_features else None,
                ai_feedback or None,
                video_width,
                video_height,
                fps,
                json.dumps(frame_data) if frame_data else None,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_user_analyses(user_id, limit=50):
    """Return list of analysis records for the user, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT a.id, a.created_at, a.original_filename, a.output_full_filename, a.output_skeleton_filename,
                      a.release_features, a.ai_feedback, a.video_width, a.video_height, a.fps, a.frame_data, a.player_id,
                      p.name as player_name, p.age, p.height, p.weight, p.bowling_arm, p.bowling_style, p.team_name, p.photo_filename
               FROM analyses a
               LEFT JOIN players p ON a.player_id = p.id
               WHERE a.user_id = ? ORDER BY a.created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    out = []
    for r in rows:
        rec = dict(r)
        if rec.get("release_features"):
            try:
                rec["release_features"] = json.loads(rec["release_features"])
            except Exception:
                rec["release_features"] = {}
        if rec.get("frame_data") is not None:
            try:
                rec["frame_data"] = json.loads(rec["frame_data"]) if rec["frame_data"] else []
            except Exception:
                rec["frame_data"] = []
        else:
            rec["frame_data"] = []
        rec["created_at"] = rec["created_at"] or ""
        out.append(rec)
    return out


def get_analysis_by_id(analysis_id, user_id):
    """Return one analysis record if it exists and belongs to the user, else None."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT a.id, a.created_at, a.original_filename, a.output_full_filename, a.output_skeleton_filename,
                      a.release_features, a.ai_feedback, a.video_width, a.video_height, a.fps, a.frame_data, a.player_id,
                      p.name as player_name, p.age, p.height, p.weight, p.bowling_arm, p.bowling_style, p.team_name, p.photo_filename
               FROM analyses a
               LEFT JOIN players p ON a.player_id = p.id
               WHERE a.id = ? AND a.user_id = ?""",
            (analysis_id, user_id),
        ).fetchone()
    if not row:
        return None
    rec = dict(row)
    if rec.get("release_features"):
        try:
            rec["release_features"] = json.loads(rec["release_features"])
        except Exception:
            rec["release_features"] = {}
    if rec.get("frame_data") is not None:
        try:
            rec["frame_data"] = json.loads(rec["frame_data"]) if rec["frame_data"] else []
        except Exception:
            rec["frame_data"] = []
    else:
        rec["frame_data"] = []
    return rec


# ==============================================================================
# PLAYER MANAGEMENT FUNCTIONS
# ==============================================================================

def create_player(user_id, name, age=None, height=None, weight=None, bowling_arm=None, 
                  bowling_style=None, team_name=None, photo_filename=None):
    """Create a new player profile. Returns (player_dict, None) or (None, error_message)."""
    if not name or not name.strip():
        return None, "Player name is required."
    
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO players (user_id, name, age, height, weight, bowling_arm, 
               bowling_style, team_name, photo_filename)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, name.strip(), age, height, weight, bowling_arm, bowling_style, team_name, photo_filename),
        )
        conn.commit()
        player_id = cur.lastrowid
    return get_player_by_id(player_id, user_id), None


def get_player_by_id(player_id, user_id):
    """Return player record if it exists and belongs to the user, else None."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT id, user_id, name, age, height, weight, bowling_arm, bowling_style, 
                      team_name, photo_filename, created_at, updated_at
               FROM players WHERE id = ? AND user_id = ?""",
            (player_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def get_user_players(user_id, limit=100):
    """Return list of all players for a user, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, user_id, name, age, height, weight, bowling_arm, bowling_style, 
                      team_name, photo_filename, created_at, updated_at
               FROM players WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def update_player(player_id, user_id, name=None, age=None, height=None, weight=None, 
                  bowling_arm=None, bowling_style=None, team_name=None, photo_filename=None):
    """Update player profile. Returns (player_dict, None) or (None, error_message)."""
    player = get_player_by_id(player_id, user_id)
    if not player:
        return None, "Player not found."
    
    updates = []
    params = []
    
    if name is not None:
        if not name.strip():
            return None, "Player name cannot be empty."
        updates.append("name = ?")
        params.append(name.strip())
    
    if age is not None:
        updates.append("age = ?")
        params.append(age)
    
    if height is not None:
        updates.append("height = ?")
        params.append(height)
    
    if weight is not None:
        updates.append("weight = ?")
        params.append(weight)
    
    if bowling_arm is not None:
        updates.append("bowling_arm = ?")
        params.append(bowling_arm)
    
    if bowling_style is not None:
        updates.append("bowling_style = ?")
        params.append(bowling_style)
    
    if team_name is not None:
        updates.append("team_name = ?")
        params.append(team_name)
    
    if photo_filename is not None:
        updates.append("photo_filename = ?")
        params.append(photo_filename)
    
    if updates:
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(player_id)
        params.append(user_id)
        with get_db() as conn:
            conn.execute(
                f"UPDATE players SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                tuple(params),
            )
            conn.commit()
    
    return get_player_by_id(player_id, user_id), None


def delete_player(player_id, user_id):
    """Delete a player. Returns (True, None) or (False, error_message)."""
    player = get_player_by_id(player_id, user_id)
    if not player:
        return False, "Player not found."
    
    with get_db() as conn:
        conn.execute("DELETE FROM players WHERE id = ? AND user_id = ?", (player_id, user_id))
        conn.commit()
    return True, None


def login_required(f):
    """Decorator: require session user. Returns 401 JSON for API-style requests, redirect for GET."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.method == "POST" or request.path.startswith("/api"):
                return jsonify({"error": "Please log in to continue.", "login_required": True}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapped


init_db()


def _ensure_pose_model():
    """Download pose landmarker model if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(POSE_MODEL_PATH):
        print("Downloading pose_landmarker model...")
        r = requests.get(POSE_MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(POSE_MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("Model downloaded.")
    return POSE_MODEL_PATH


def _create_pose_landmarker():
    """Create PoseLandmarker (video mode) for processing frames."""
    model_path = _ensure_pose_model()
    base_options = base_options_lib.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


# Pose landmark indices (MediaPipe standard)
_LANDMARK_LEFT_SHOULDER, _LANDMARK_RIGHT_SHOULDER = 11, 12
_LANDMARK_LEFT_ELBOW, _LANDMARK_LEFT_WRIST = 13, 15
_LANDMARK_LEFT_HIP, _LANDMARK_RIGHT_HIP = 23, 24
_LANDMARK_LEFT_KNEE, _LANDMARK_LEFT_ANKLE, _LANDMARK_RIGHT_ANKLE = 25, 27, 28


# ==============================================================================
# 2. BACKEND: POSE ANALYSIS & AI FEEDBACK LOGIC
# ==============================================================================

# === AI Feedback Setup (Modified to return data, not print) ===
def deepseek_feedback(features):
    """
    Sends pose features to the DeepSeek model via OpenRouter for feedback.
    Returns the AI's feedback as a string.
    """
    if not API_KEY or "sk-or-" not in API_KEY:
        print("API key not configured.")
        return "API key not configured. Please check the server."

    # --- FIX: Changed prompt to request 4 tips instead of 2 ---
    prompt = f"""
You are a professional cricket biomechanics coach. Based on this data from a bowling action's release point:
- Elbow Angle: {features['elbow_angle']}°
- Shoulder Angle: {features['shoulder_angle']}°
- Arm Verticality: {features['arm_verticality']}°
- Stride Length (normalized to shoulder width): {features['stride_length']}×

Give 4 very short, direct, and actionable tips (under 20 words each) to improve the bowling form, focusing on different aspects like balance, power, injury prevention, and efficiency.
Format the output as a simple list. Example:
- Keep your elbow straighter for more power.
- Shorten your stride to improve balance.
- Engage your core for better stability.
- Follow through fully towards the target.
"""

    # send the prompt to OpenRouter (DeepSeek) similar to training plan
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "ApexCric"
    }
    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"]
            return reply.strip()
        else:
            error_message = f"API error {res.status_code}: {res.text}"
            print(error_message)
            return ""  # return empty so UI shows fallback
    except Exception as e:
        print(f"AI feedback request failed: {e}")
        return ""

def generate_training_plan(features, goal):
    """Create a weekly training plan based on biomechanical features and a training goal."""
    if not API_KEY or "sk-or-" not in API_KEY:
        print("API key not configured.")
        return "API key not configured. Please check the server."

    prompt = f"""
You are a certified cricket strength and conditioning coach. Given the following biomechanical metrics from a bowling release:
- Elbow Angle: {features.get('elbow_angle')}°
- Shoulder Angle: {features.get('shoulder_angle')}°
- Arm Verticality: {features.get('arm_verticality')}°
- Stride Length: {features.get('stride_length')}×

Create a ** simple, beginner-friendly 2‑week training plan** focused on **{goal}**. Use clear everyday language and basic drills that require minimal equipment. Format the response in plain Markdown with:

1. A short intro explaining the main goal in 3 or more sentence.
2. A heading for each day (e.g. `### Day 1`) followed by 3‑4 easy bullet points describing the exercises.
3. Exactly 14 day sections – keep each day very concise.
4. Use `-` for list items, headings with `###`.
5. Respond only with the Markdown text so it can be displayed directly.

Example:

### Day 1
- Walk briskly 10 minutes
- Shoulder rolls x20
- Simple wrist stretches

### Day 2
- Gentle jogging 5 minutes

...and so on until Day 14.

"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "ApexCric"
    }

    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"]
            return reply.strip()
        else:
            error_message = f"API error {res.status_code}: {res.text}"
            print(f" {error_message}")
            return f"Could not get training plan. {error_message}"
    except Exception as e:
        error_message = f"API request failed: {str(e)}"
        print(f" {error_message}")
        return f"Could not get training plan. {error_message}"


def generate_tactical_advice(bowler_type, bowler_hand, batsman_hand, match_phase, match_format):
    """Generate tactical cricket bowling advice using AI."""
    if not API_KEY or "sk-or-" not in API_KEY:
        return "API key not configured."

    prompt = f"""
You are an expert cricket tactician and field strategist. Generate a detailed tactical bowling plan with the following context:
- Bowler Type: {bowler_type}
- Bowler Hand: {bowler_hand}
- Batsman Hand: {batsman_hand}
- Match Phase: {match_phase}
- Match Format: {match_format}

Provide a comprehensive tactical card with these sections (use exact headings with ###):

### Bowling Lengths
- (List 3-4 recommended lengths with percentages)

### Line Strategy
- (List 3-4 lines with percentages)

### Field Setup
- Slip, Gully, Point, Cover, Mid-off, Mid-on, Square Leg, Fine Leg, Deep Square, Long-on, Long-off

### Ball-by-Ball Plan
- Ball 1: (length and line)
- Ball 2: (length and line)
- Ball 3: (length and line)
- Ball 4: (length and line)
- Ball 5: (length and line)
- Ball 6: (length and line)

### Key Strategies
- Wicket-taking approach: (1 line)
- Run containment: (1 line)
- Variations to use: (1 line)

### Delivery Mix (%)
- (Show percentage breakdown for this phase)

Use clear, tactical language. No extra text outside these sections.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "ApexCric"
    }

    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"]
            raw = reply.strip()

            # Convert the AI markdown-style output into two formats:
            # 1) HTML for web display (centered, highlighted headings, styled lists)
            # 2) Plain text fallback (keeps original markdown-like style) for PDFs

            lines = raw.splitlines()
            html_lines = []
            text_lines = []
            in_list = False
            for ln in lines:
                s = ln.strip()
                if s.startswith('###'):
                    # close any open list
                    if in_list:
                        html_lines.append('</ul>')
                        in_list = False
                    heading = s.replace('###', '').strip()
                    # center and highlight the heading
                    html_lines.append(f'<h3 style="text-align:center; color:#0ea8f0; font-family:Orbitron, sans-serif; margin:16px 0;">{heading}</h3>')
                    text_lines.append('### ' + heading)
                elif s.startswith('-'):
                    item = s.lstrip('-').strip()
                    if not in_list:
                        html_lines.append('<ul style="list-style:none; padding:0; margin:8px 0; display:flex; flex-direction:column; gap:6px; align-items:center;">')
                        in_list = True
                    # make each point a styled pill-like item
                    html_lines.append(f'<li style="background:linear-gradient(90deg,#071228,#06202b); color:#e6fff9; padding:10px 14px; border-radius:999px; box-shadow:0 2px 8px rgba(0,0,0,0.4); max-width:820px;">{item}</li>')
                    text_lines.append(f'- {item}')
                elif s == '':
                    # blank line
                    if in_list:
                        html_lines.append('</ul>')
                        in_list = False
                    html_lines.append('<br/>')
                    text_lines.append('')
                else:
                    # plain paragraph
                    if in_list:
                        html_lines.append('</ul>')
                        in_list = False
                    html_lines.append(f'<p style="color:#cfeff4; max-width:820px; margin:6px auto; text-align:center;">{s}</p>')
                    text_lines.append(s)

            if in_list:
                html_lines.append('</ul>')

            html_formatted = '\n'.join(html_lines)
            text_fallback = '\n'.join(text_lines)

            # Return a small wrapper that includes both formats so callers can choose.
            # Format: special delimiter '---HTML---' then HTML, then '---TEXT---' then plain text.
            # Existing callers that expect plain text can split on the delimiter or ignore HTML part.
            combined = f"---HTML---\n{html_formatted}\n---TEXT---\n{text_fallback}"
            return combined
        else:
            return f"Could not generate tactical advice. Error: {res.status_code}"
    except Exception as e:
        return f"Error generating tactical advice: {str(e)}"


def generate_pdf_tactical_card(bowler_type, bowler_hand, batsman_hand, match_phase, match_format, tactical_advice):
    """Generate a medium-complexity PDF for the tactical card."""
    # If tactical_advice is the combined format returned by generate_tactical_advice,
    # extract the plain-text portion after the '---TEXT---' delimiter so existing
    # PDF-generation logic continues to work.
    if isinstance(tactical_advice, str) and '---TEXT---' in tactical_advice:
        try:
            tactical_advice = tactical_advice.split('---TEXT---', 1)[1].strip()
        except Exception:
            # fallback to original if anything unexpected
            pass
    pdf_filename = f"tactical_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#19faaa'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#0ea8f0'),
        spaceAfter=8,
        spaceBefore=8
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=9,
        leading=12
    )
    
    # Title
    story.append(Paragraph("🏏 TACTICAL BOWLING CARD", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Match Context Table
    context_data = [
        ['MATCH CONTEXT', ''],
        ['Bowler Type', bowler_type],
        ['Bowler Hand', bowler_hand],
        ['Batsman Hand', batsman_hand],
        ['Match Phase', match_phase],
        ['Match Format', match_format],
    ]
    context_table = Table(context_data, colWidths=[2*inch, 3.5*inch])
    context_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#0ea8f0')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    story.append(context_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Tactical Advice Content
    advice_lines = tactical_advice.split('\n')
    for line in advice_lines:
        line = line.strip()
        if line.startswith('###'):
            heading = line.replace('###', '').strip()
            story.append(Paragraph(heading, heading_style))
        elif line.startswith('-'):
            bullet = line.replace('-', '', 1).strip()
            story.append(Paragraph(f"• {bullet}", body_style))
        elif line:
            story.append(Paragraph(line, body_style))
        else:
            story.append(Spacer(1, 0.05*inch))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=7,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", footer_style))
    
    # Build PDF
    doc.build(story)
    return pdf_filename


def create_share_link(user_id, plan_type, plan_data, expires_days=7):
    """Create a shareable link for a tactical/training plan."""
    share_id = secrets.token_urlsafe(16)
    expires_at = datetime.now() + timedelta(days=expires_days)
    
    with get_db() as conn:
        conn.execute(
            """INSERT INTO shared_plans (user_id, share_id, plan_type, plan_data, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, share_id, plan_type, json.dumps(plan_data), expires_at.isoformat())
        )
        conn.commit()
    
    return share_id, expires_at


def get_shared_plan(share_id):
    """Retrieve a shared plan (no login required)."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT user_id, plan_type, plan_data, expires_at, access_count
               FROM shared_plans WHERE share_id = ?""",
            (share_id,)
        ).fetchone()
    
    if not row:
        return None
    
    rec = dict(row)
    expires_at = datetime.fromisoformat(rec['expires_at'])
    
    if expires_at < datetime.now():
        # Delete expired share
        with get_db() as conn:
            conn.execute("DELETE FROM shared_plans WHERE share_id = ?", (share_id,))
            conn.commit()
        return None
    
    # Update access count
    with get_db() as conn:
        conn.execute(
            "UPDATE shared_plans SET access_count = access_count + 1 WHERE share_id = ?",
            (share_id,)
        )
        conn.commit()
    
    rec['plan_data'] = json.loads(rec['plan_data'])
    return rec


def delete_share(share_id, user_id):
    """Delete a share link (auth required)."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM shared_plans WHERE share_id = ? AND user_id = ?",
            (share_id, user_id)
        )
        conn.commit()


    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:5000",
    "X-Title": "ApexCric"
    }

    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"]
            return reply.strip()
        else:
            error_message = f"API error {res.status_code}: {res.text}"
            print(f" {error_message}")
            return f"Could not get AI feedback. {error_message}"
    except Exception as e:
        error_message = f"API request failed: {str(e)}"
        print(f" {error_message}")
        return f"Could not get AI feedback. {error_message}"


def calculate_3d_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def _visibility(lm):
    """Get visibility score for a NormalizedLandmark (Tasks API)."""
    v = getattr(lm, 'visibility', None)
    if v is not None:
        return v
    return getattr(lm, 'presence', 1.0) or 1.0


def extract_features(landmarks_list):
    """Extracts key biomechanical features from pose landmarks (Tasks API: list of NormalizedLandmark)."""
    try:
        if len(landmarks_list) < 33:
            return None
        ls = [landmarks_list[_LANDMARK_LEFT_SHOULDER].x, landmarks_list[_LANDMARK_LEFT_SHOULDER].y, landmarks_list[_LANDMARK_LEFT_SHOULDER].z]
        le = [landmarks_list[_LANDMARK_LEFT_ELBOW].x, landmarks_list[_LANDMARK_LEFT_ELBOW].y, landmarks_list[_LANDMARK_LEFT_ELBOW].z]
        lw = [landmarks_list[_LANDMARK_LEFT_WRIST].x, landmarks_list[_LANDMARK_LEFT_WRIST].y, landmarks_list[_LANDMARK_LEFT_WRIST].z]
        lh = [landmarks_list[_LANDMARK_LEFT_HIP].x, landmarks_list[_LANDMARK_LEFT_HIP].y, landmarks_list[_LANDMARK_LEFT_HIP].z]
        lk = [landmarks_list[_LANDMARK_LEFT_KNEE].x, landmarks_list[_LANDMARK_LEFT_KNEE].y, landmarks_list[_LANDMARK_LEFT_KNEE].z]
        la = [landmarks_list[_LANDMARK_LEFT_ANKLE].x, landmarks_list[_LANDMARK_LEFT_ANKLE].y, landmarks_list[_LANDMARK_LEFT_ANKLE].z]
        ra = [landmarks_list[_LANDMARK_RIGHT_ANKLE].x, landmarks_list[_LANDMARK_RIGHT_ANKLE].y, landmarks_list[_LANDMARK_RIGHT_ANKLE].z]
        rs = [landmarks_list[_LANDMARK_RIGHT_SHOULDER].x, landmarks_list[_LANDMARK_RIGHT_SHOULDER].y, landmarks_list[_LANDMARK_RIGHT_SHOULDER].z]

        vs = [
            _visibility(landmarks_list[_LANDMARK_LEFT_SHOULDER]),
            _visibility(landmarks_list[_LANDMARK_LEFT_ELBOW]),
            _visibility(landmarks_list[_LANDMARK_LEFT_WRIST]),
            _visibility(landmarks_list[_LANDMARK_LEFT_HIP]),
            _visibility(landmarks_list[_LANDMARK_LEFT_ANKLE]),
            _visibility(landmarks_list[_LANDMARK_RIGHT_ANKLE]),
        ]
        if min(vs) < 0.6:
            return None

        shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
        return {
            "elbow_angle": round(calculate_3d_angle(ls, le, lw), 1),
            "shoulder_angle": round(calculate_3d_angle(lh, ls, le), 1),
            "arm_verticality": round(calculate_3d_angle(lh, ls, lw), 1),
            "front_knee_angle": round(calculate_3d_angle(lh, lk, la), 1),
            "stride_length": round(np.linalg.norm(np.array(la) - np.array(ra)) / shoulder_width if shoulder_width > 0 else 0, 2),
        }
    except Exception:
        return None


def _draw_landmarks_on_frame(frame, landmarks_list, h, w, features=None):
    """Draw pose skeleton on frame from list of NormalizedLandmark (normalized 0-1)."""
    if not landmarks_list or len(landmarks_list) < 33:
        return
    pts = []
    for lm in landmarks_list:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        i, j = conn.start, conn.end
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (25, 250, 170), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
    if features:
        draw_info_box(frame, features)

def draw_info_box(frame, features):
    """Draws the feedback box with release metrics on the video frame."""
    h, w, _ = frame.shape
    box_x, box_y, box_h = 10, h - 130, 120
    cv2.rectangle(frame, (box_x, box_y), (box_x + 280, box_y + box_h), (20, 20, 20), -1)
    
    cv2.putText(frame, "Metrics (at Release)", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    metrics = [
        f"Elbow Angle: {features.get('elbow_angle', 'N/A')}°",
        f"Arm Verticality: {features.get('arm_verticality', 'N/A')}°",
        f"Knee Angle: {features.get('front_knee_angle', 'N/A')}°",
        f"Stride (norm): {features.get('stride_length', 'N/A')}x"
    ]
    for i, metric in enumerate(metrics):
        cv2.putText(frame, metric, (box_x + 10, box_y + 45 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

def analyze_bowling_pose(video_path):
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_duration_ms = int(1000.0 / fps)

    pose_landmarker = _create_pose_landmarker()

    # Pass 1: Extract data from all frames and find release point
    all_frame_features = []
    release_frame_features = None
    max_elbow_angle = 0
    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        mp_image = Image(ImageFormat.SRGB, rgb)
        result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_duration_ms

        frame_features = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            frame_features = extract_features(result.pose_landmarks[0])
            if frame_features and frame_features.get('elbow_angle', 0) > max_elbow_angle:
                max_elbow_angle = frame_features['elbow_angle']
                release_frame_features = frame_features
        all_frame_features.append(frame_features)

    ai_feedback = "No suitable pose detected for analysis."
    if release_frame_features:
        ai_feedback = deepseek_feedback(release_frame_features)

    pose_landmarker.close()

    # Pass 2: Generate both Full and Skeleton videos (new landmarker so timestamps can start at 0)
    pose_landmarker_2 = _create_pose_landmarker()
    uid = uuid.uuid4().hex
    output_full_filename = f"output_full_{uid}.mp4"
    output_skeleton_filename = f"output_skeleton_{uid}.mp4"
    output_full_path = os.path.join(app.config['STATIC_FOLDER'], output_full_filename)
    output_skeleton_path = os.path.join(app.config['STATIC_FOLDER'], output_skeleton_filename)

    writer_full = cv2.VideoWriter(output_full_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    writer_skeleton = cv2.VideoWriter(output_skeleton_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    timestamp_ms = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags['C_CONTIGUOUS']:
            rgb = np.ascontiguousarray(rgb)
        mp_image = Image(ImageFormat.SRGB, rgb)
        result = pose_landmarker_2.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += frame_duration_ms

        black_frame = np.zeros_like(frame)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = result.pose_landmarks[0]
            _draw_landmarks_on_frame(frame, lm_list, h, w, release_frame_features)
            _draw_landmarks_on_frame(black_frame, lm_list, h, w, release_frame_features)

        writer_full.write(frame)
        writer_skeleton.write(black_frame)

    cap.release()
    writer_full.release()
    writer_skeleton.release()
    pose_landmarker_2.close()

    print("Analysis complete! Outputs generated.")

    return {
        "output_video_full_url": url_for('static', filename=output_full_filename, _external=True),
        "output_video_skeleton_url": url_for('static', filename=output_skeleton_filename, _external=True),
        "output_full_filename": output_full_filename,
        "output_skeleton_filename": output_skeleton_filename,
        "release_features": release_frame_features if release_frame_features else {},
        "frame_data": all_frame_features,
        "ai_feedback": ai_feedback,
        "video_dimensions": {"width": w, "height": h},
        "fps": fps
    }

# ==============================================================================
# 3. FLASK ROUTES (templates in templates/)
# ==============================================================================

@app.route('/')
def index():
    """Redirect to dashboard if logged in, else to login."""
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route('/login')
def login_page():
    """Login/signup page. Redirect to dashboard if already logged in."""
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard: upload + history. Requires login."""
    user = get_user_by_id(session["user_id"])
    return render_template(
        "dashboard.html",
        logged_in=True,
        standalone_upload=True,
        user_email=user["email"],
        user_name=user.get("name") or user["email"],
    )


@app.route('/players')
@login_required
def players_page():
    """Players management page. Requires login."""
    user = get_user_by_id(session["user_id"])
    return render_template(
        "players.html",
        logged_in=True,
        user_email=user["email"],
        user_name=user.get("name") or user["email"],
    )


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """Edit profile page. GET: show form. POST: update name, email, or password."""
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    if request.method == 'GET':
        return render_template(
            "profile.html",
            logged_in=True,
            user_email=user["email"],
            user_name=user.get("name") or "",
        )
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip() or None
    email = (data.get("email") or "").strip().lower() or None
    new_password = (data.get("new_password") or "").strip() or None
    current_password = (data.get("current_password") or "") if new_password else None
    updated_user, err = update_user(
        user_id,
        name=name if name else user.get("name"),
        email=email if email else user["email"],
        new_password=new_password,
        current_password=current_password,
    )
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({
        "ok": True,
        "email": updated_user["email"],
        "name": updated_user.get("name") or updated_user["email"],
    })


@app.route('/analysis/<int:analysis_id>')
@login_required
def analysis_page(analysis_id):
    """View a single analysis. Requires login and ownership."""
    user_id = session.get("user_id")
    analysis = get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return "Analysis not found", 404
    base = request.url_root.rstrip("/")
    analysis["output_video_full_url"] = f"{base}/static/{analysis['output_full_filename']}"
    analysis["output_video_skeleton_url"] = f"{base}/static/{analysis['output_skeleton_filename']}"
    user = get_user_by_id(user_id)
    return render_template(
        "analysis_page.html",
        analysis=analysis,
        logged_in=True,
        user_email=user["email"],
        user_name=user.get("name") or user["email"],
    )


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign up: create account and log in (API)."""
    if request.method == 'GET':
        return redirect(url_for('login_page'))
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''
    name = (data.get('name') or '').strip()
    user, err = create_user(email, password, name=name or None)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    token = create_auth_token(user["id"])
    return jsonify({"ok": True, "email": user["email"], "token": token})


@app.route('/login', methods=['POST'])
def login():
    """Log in with email and password (API)."""
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    user = get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "Invalid email or password."}), 401
    session.clear()
    session["user_id"] = user["id"]
    session.permanent = True
    token = create_auth_token(user["id"])
    return jsonify({"ok": True, "email": user["email"], "token": token})


@app.route('/api/session/restore', methods=['POST'])
def session_restore():
    """Restore session from a stored auth token (e.g. after server restart)."""
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    user_id = lookup_user_by_token(token)
    if not user_id:
        return jsonify({"ok": False, "error": "Invalid or expired token"}), 401
    session.clear()
    session["user_id"] = user_id
    session.permanent = True
    return jsonify({"ok": True})


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for('login_page'))


# serve uploaded media (player photos, etc.)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Return a file from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/history')
@login_required
def api_history():
    """Return analysis history for the current user (JSON)."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in", "login_required": True}), 401
    analyses = get_user_analyses(user_id)
    # Build static URLs for each (frontend can use same origin)
    base = request.url_root.rstrip("/")
    for a in analyses:
        a["output_video_full_url"] = f"{base}/static/{a['output_full_filename']}"
        a["output_video_skeleton_url"] = f"{base}/static/{a['output_skeleton_filename']}"
    return jsonify({"analyses": analyses})


@app.route('/api/training_plan', methods=['POST'])
@login_required
def api_training_plan():
    """Generate a training plan from an analysis and a selected goal."""
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    goal = (data.get("goal") or "").strip()
    if not analysis_id or not goal:
        return jsonify({"ok": False, "error": "analysis_id and goal are required"}), 400
    user_id = session.get("user_id")
    analysis = get_analysis_by_id(analysis_id, user_id)
    if not analysis:
        return jsonify({"ok": False, "error": "Analysis not found"}), 404
    features = analysis.get("release_features") or {}
    if not features:
        return jsonify({"ok": False, "error": "No release features available for this analysis"}), 400
    plan = generate_training_plan(features, goal)
    return jsonify({"ok": True, "plan": plan})


@app.route('/api/tactical_advice', methods=['POST'])
def api_tactical_advice():
    """Generate tactical bowling advice (no login required)."""
    data = request.get_json(silent=True) or {}
    bowler_type = (data.get("bowler_type") or "").strip()
    bowler_hand = (data.get("bowler_hand") or "").strip()
    batsman_hand = (data.get("batsman_hand") or "").strip()
    match_phase = (data.get("match_phase") or "").strip()
    match_format = (data.get("match_format") or "").strip()
    
    if not all([bowler_type, bowler_hand, batsman_hand, match_phase, match_format]):
        return jsonify({"ok": False, "error": "All fields are required"}), 400
    
    advice = generate_tactical_advice(bowler_type, bowler_hand, batsman_hand, match_phase, match_format)

    # If the generator returned the combined wrapper, split into HTML and TEXT parts
    advice_html = None
    advice_text = None
    if isinstance(advice, str) and '---HTML---' in advice and '---TEXT---' in advice:
        try:
            parts = advice.split('---TEXT---', 1)
            html_part = parts[0].replace('---HTML---', '').strip()
            text_part = parts[1].strip()
            advice_html = html_part
            advice_text = text_part
        except Exception:
            advice_text = advice
    else:
        advice_text = advice

    # Return both formats; keep `advice` as the plain-text fallback for compatibility
    return jsonify({"ok": True, "advice": advice_text, "advice_html": advice_html, "advice_text": advice_text})


@app.route('/api/export_tactical_pdf', methods=['POST'])
@login_required
def api_export_tactical_pdf():
    """Generate and share a PDF tactical card."""
    data = request.get_json(silent=True) or {}
    bowler_type = (data.get("bowler_type") or "").strip()
    bowler_hand = (data.get("bowler_hand") or "").strip()
    batsman_hand = (data.get("batsman_hand") or "").strip()
    match_phase = (data.get("match_phase") or "").strip()
    match_format = (data.get("match_format") or "").strip()
    tactical_advice = (data.get("tactical_advice") or "").strip()
    expires_days = data.get("expires_days", 7)
    
    if not all([bowler_type, bowler_hand, batsman_hand, match_phase, match_format, tactical_advice]):
        return jsonify({"ok": False, "error": "All fields are required"}), 400
    
    user_id = session.get("user_id")
    
    try:
        pdf_filename = generate_pdf_tactical_card(bowler_type, bowler_hand, batsman_hand, match_phase, match_format, tactical_advice)
        
        plan_data = {
            "bowler_type": bowler_type,
            "bowler_hand": bowler_hand,
            "batsman_hand": batsman_hand,
            "match_phase": match_phase,
            "match_format": match_format,
            "tactical_advice": tactical_advice,
            "pdf_filename": pdf_filename
        }
        
        share_id, expires_at = create_share_link(user_id, "tactical", plan_data, expires_days)
        share_url = url_for('view_shared_plan', share_id=share_id, _external=True)
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(share_url)
        qr.make()
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for embedding
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        qr_base64 = base64.b64encode(img_io.getvalue()).decode()
        
        return jsonify({
            "ok": True,
            "share_url": share_url,
            "expires_at": expires_at.isoformat(),
            "qr_code": f"data:image/png;base64,{qr_base64}",
            "pdf_download": url_for('uploaded_file', filename=pdf_filename, _external=True)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/share/<share_id>')
def view_shared_plan(share_id):
    """View a shared tactical/training plan."""
    plan = get_shared_plan(share_id)
    if not plan:
        return "Shared plan not found or has expired.", 404
    
    return jsonify({
        "ok": True,
        "plan_type": plan['plan_type'],
        "plan_data": plan['plan_data'],
        "access_count": plan['access_count']
    })


@app.route('/api/download_tactical_pdf', methods=['POST'])
@login_required
def api_download_tactical_pdf():
    """Generate a tactical PDF and return it as a file download."""
    data = request.get_json(silent=True) or {}
    bowler_type = (data.get("bowler_type") or "").strip()
    bowler_hand = (data.get("bowler_hand") or "").strip()
    batsman_hand = (data.get("batsman_hand") or "").strip()
    match_phase = (data.get("match_phase") or "").strip()
    match_format = (data.get("match_format") or "").strip()
    tactical_advice = (data.get("tactical_advice") or "").strip()

    if not all([bowler_type, bowler_hand, batsman_hand, match_phase, match_format, tactical_advice]):
        return jsonify({"ok": False, "error": "All fields are required"}), 400

    try:
        pdf_filename = generate_pdf_tactical_card(bowler_type, bowler_hand, batsman_hand, match_phase, match_format, tactical_advice)
        return send_from_directory(app.config['UPLOAD_FOLDER'], pdf_filename, as_attachment=True)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/delete_share/<share_id>', methods=['DELETE'])
@login_required
def api_delete_share(share_id):
    """Delete a shared link."""
    user_id = session.get("user_id")
    try:
        delete_share(share_id, user_id)
        return jsonify({"ok": True, "message": "Share deleted"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Handles video upload, processing, and returns JSON results. Requires login."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = f"input_{uuid.uuid4().hex}.mp4"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        try:
            results = analyze_bowling_pose(input_path)
            if "error" not in results:
                dims = results.get("video_dimensions") or {}
                player_id = request.form.get("player_id", type=int)
                analysis_id = save_analysis(
                    session["user_id"],
                    file.filename,
                    results.get("output_full_filename", ""),
                    results.get("output_skeleton_filename", ""),
                    results.get("release_features"),
                    results.get("ai_feedback"),
                    dims.get("width") or 0,
                    dims.get("height") or 0,
                    results.get("fps") or 30,
                    results.get("frame_data"),
                    player_id=player_id,
                )
                results["analysis_id"] = analysis_id
            return jsonify(results)
        except Exception as e:
            print(f"Error during analysis route: {e}")
            return jsonify({"error": "An internal error occurred during analysis."}), 500
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)


# ==============================================================================
# PLAYER API ROUTES
# ==============================================================================

@app.route('/api/players', methods=['GET', 'POST'])
@login_required
def api_players():
    """GET: List all players. POST: Create a new player."""
    user_id = session.get("user_id")
    
    if request.method == 'GET':
        players = get_user_players(user_id)
        return jsonify({"ok": True, "players": players})
    
    # POST: Create new player
    data = request.form
    name = (data.get("name") or "").strip()
    age = data.get("age", type=int)
    height = data.get("height", type=float)
    weight = data.get("weight", type=float)
    bowling_arm = (data.get("bowling_arm") or "").strip() or None
    bowling_style = (data.get("bowling_style") or "").strip() or None
    team_name = (data.get("team_name") or "").strip() or None
    
    photo_filename = None
    if 'photo' in request.files:
        photo_file = request.files['photo']
        if photo_file and photo_file.filename != '':
            photo_filename = f"player_{uuid.uuid4().hex}_{photo_file.filename}"
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
            photo_file.save(photo_path)
    
    player, err = create_player(
        user_id, 
        name=name,
        age=age,
        height=height,
        weight=weight,
        bowling_arm=bowling_arm,
        bowling_style=bowling_style,
        team_name=team_name,
        photo_filename=photo_filename,
    )
    
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "player": player})


@app.route('/api/players/<int:player_id>', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_player_detail(player_id):
    """GET: Get player details. POST: Update player. DELETE: Delete player."""
    user_id = session.get("user_id")
    
    if request.method == 'GET':
        player = get_player_by_id(player_id, user_id)
        if not player:
            return jsonify({"ok": False, "error": "Player not found"}), 404
        return jsonify({"ok": True, "player": player})
    
    elif request.method == 'POST':
        data = request.form
        name = (data.get("name") or "").strip() or None
        age = data.get("age", type=int) if data.get("age") else None
        height = data.get("height", type=float) if data.get("height") else None
        weight = data.get("weight", type=float) if data.get("weight") else None
        bowling_arm = (data.get("bowling_arm") or "").strip() or None
        bowling_style = (data.get("bowling_style") or "").strip() or None
        team_name = (data.get("team_name") or "").strip() or None
        
        photo_filename = None
        if 'photo' in request.files:
            photo_file = request.files['photo']
            if photo_file and photo_file.filename != '':
                photo_filename = f"player_{uuid.uuid4().hex}_{photo_file.filename}"
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
                photo_file.save(photo_path)
        
        player, err = update_player(
            player_id, 
            user_id,
            name=name,
            age=age,
            height=height,
            weight=weight,
            bowling_arm=bowling_arm,
            bowling_style=bowling_style,
            team_name=team_name,
            photo_filename=photo_filename,
        )
        
        if err:
            return jsonify({"ok": False, "error": err}), 400
        return jsonify({"ok": True, "player": player})
    
    elif request.method == 'DELETE':
        success, err = delete_player(player_id, user_id)
        if not success:
            return jsonify({"ok": False, "error": err}), 404
        return jsonify({"ok": True, "message": "Player deleted successfully"})


# ==============================================================================
# 5. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5001)

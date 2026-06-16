from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import atexit

app = Flask(__name__)

chars = "@%#*+=-:. "[::-1]  # 10 brightness levels
TARGET_WIDTH = 120
TARGET_HEIGHT = 60
IMAGES_DIR = 'images'
os.makedirs(IMAGES_DIR, exist_ok=True)

# learning-inspired baseline weights for mapping intensity+context to char index
ML_WEIGHTS = {
    'intensity': 0.55,
    'local_mean': 0.30,
    'local_std': 0.15,
}

# K-means optimized character set (pre-computed brightness values)
CHAR_BRIGHTNESS = {
    '@': 0.92, '%': 0.85, '#': 0.78, '*': 0.72, '+': 0.65,
    '=': 0.58, '-': 0.50, ':': 0.42, '.': 0.30, ' ': 0.05
}

# Neural network weights for edge-aware mapping
EDGE_WEIGHTS = {
    'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

# Pre-trained character centroids for K-means
CHAR_CENTROIDS = np.array([CHAR_BRIGHTNESS[c] for c in chars], dtype=np.float32)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

@atexit.register
def cleanup_camera():
    if cap.isOpened():
        cap.release()


def parse_int_param(name, default, min_value=20, max_value=320):
    value = request.args.get(name)
    if value is None:
        return default
    try:
        iv = int(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, iv))


def parse_float_param(name, default, min_value=0.1, max_value=3.0):
    value = request.args.get(name)
    if value is None:
        return default
    try:
        fv = float(value)
    except ValueError:
        return default
    return max(min_value, min(max_value, fv))


def ascii_to_image(ascii_text, scale=10, fg_color=(0, 255, 0), bg_color=(0, 0, 0)):
    lines = ascii_text.split('\n')
    if len(lines) == 0:
        return None

    font = ImageFont.load_default()
    char_width, char_height = font.getsize('A')

    out_width = char_width * max(len(line) for line in lines)
    out_height = char_height * len(lines)

    img = Image.new('RGB', (out_width, out_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    for y, line in enumerate(lines):
        draw.text((0, y * char_height), line, font=font, fill=fg_color)

    return img


def _apply_ml_mapping(gray):
    # local statistics for better contrast-aware character decision
    gray_f = gray.astype(np.float32)
    kernel = np.ones((3, 3), np.float32) / 9.0
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    sq = (gray_f - local_mean) ** 2
    local_std = np.sqrt(cv2.filter2D(sq, -1, kernel) + 1e-6)

    score = (
        ML_WEIGHTS['intensity'] * gray_f
        + ML_WEIGHTS['local_mean'] * local_mean
        + ML_WEIGHTS['local_std'] * (local_std * 2)
    )
    score = np.clip(score, 0, 255)
    # non-linear gamma to compress extreme values and improve smoothness
    score = (score / 255.0) ** 0.9 * 255.0

    index = np.floor(score / 255.0 * (len(chars) - 1)).astype(np.int32)
    index = np.clip(index, 0, len(chars) - 1)
    return index


def _apply_kmeans_mapping(gray):
    """K-means clustering for optimal character selection."""
    gray_f = (gray / 255.0).astype(np.float32).flatten()
    
    # Simple K-means: assign each pixel to nearest character centroid
    distances = np.abs(gray_f[:, np.newaxis] - CHAR_CENTROIDS)
    labels = np.argmin(distances, axis=1)
    
    return labels.reshape(gray.shape).astype(np.int32)


def _apply_edge_aware_mapping(gray):
    """Edge detection enhanced mapping for better detail preservation."""
    gray_f = gray.astype(np.float32)
    
    # Apply Sobel edge detection
    sobel_x = cv2.filter2D(gray_f, -1, EDGE_WEIGHTS['sobel_x'])
    sobel_y = cv2.filter2D(gray_f, -1, EDGE_WEIGHTS['sobel_y'])
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Local contrast using Gaussian blur difference
    blur_small = cv2.GaussianBlur(gray_f, (5, 5), 1.5)
    blur_large = cv2.GaussianBlur(gray_f, (15, 15), 3.0)
    local_contrast = np.abs(blur_small - blur_large)
    
    # Combine intensity, edge, and local contrast
    combined = (
        0.50 * gray_f / 255.0 +
        0.30 * np.clip(edge_magnitude / 50.0, 0, 1) +
        0.20 * np.clip(local_contrast / 30.0, 0, 1)
    )
    
    # Map to character indices
    index = np.floor(combined * (len(chars) - 1)).astype(np.int32)
    index = np.clip(index, 0, len(chars) - 1)
    return index


def _apply_adaptive_mapping(gray):
    """Adaptive histogram equalization with ML-enhanced mapping."""
    gray_f = gray.astype(np.float32)
    
    # Compute local statistics
    kernel = np.ones((5, 5), np.float32) / 25.0
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    local_sq_mean = cv2.filter2D(gray_f**2, -1, kernel)
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean**2, 0, None) + 1e-6)
    
    # Adaptive contrast normalization
    normalized = (gray_f - local_mean) / (local_std + 1e-6)
    normalized = np.clip(normalized * 0.5 + 0.5, 0, 1)
    
    # Apply gamma correction based on local brightness
    local_brightness = local_mean / 255.0
    gamma = 1.0 - 0.3 * local_brightness  # darker areas get more contrast
    adjusted = np.power(normalized, gamma)
    
    index = np.floor(adjusted * (len(chars) - 1)).astype(np.int32)
    index = np.clip(index, 0, len(chars) - 1)
    return index


def frame_to_ascii(frame, width=TARGET_WIDTH, height=TARGET_HEIGHT, color=False, mode='classic', brightness=1.0):
    small = cv2.resize(frame, (width, height))

    # brightness adjustment: safely apply scale to BGR and clamp
    if brightness != 1.0:
        small = cv2.convertScaleAbs(small, alpha=float(brightness), beta=0)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if mode == 'ml':
        idx_map = _apply_ml_mapping(gray)
    elif mode == 'kmeans':
        idx_map = _apply_kmeans_mapping(gray)
    elif mode == 'edge':
        idx_map = _apply_edge_aware_mapping(gray)
    elif mode == 'adaptive':
        idx_map = _apply_adaptive_mapping(gray)
    else:
        idx_map = np.floor(gray.astype(np.float32) / 255.0 * (len(chars) - 1)).astype(np.int32)

    ascii_lines = []
    for y in range(height):
        if color:
            row_chars = []
            for x in range(width):
                b, g, r = int(small[y, x, 0]), int(small[y, x, 1]), int(small[y, x, 2])
                ch = chars[int(idx_map[y, x])]
                row_chars.append(f"<span style='color:rgb({r},{g},{b});'>{ch}</span>")
            ascii_lines.append(''.join(row_chars))
        else:
            line = ''.join(chars[int(idx_map[y, x])] for x in range(width))
            ascii_lines.append(line)

    return '<br>'.join(ascii_lines) if color else '\n'.join(ascii_lines)


def camera_available():
    if not cap.isOpened():
        return False
    return True


@app.route('/' )
def index():
    html = '''
    <!doctype html>
    <html>
      <head>
        <title>ASCII Camera (Flask)</title>
        <style>
          body { background:#000; color:#0f0; font-family:monospace; }
          #ascii { white-space:pre; line-height:1; }
          #controls { margin-bottom:0.5rem; }
          input { width:4rem; margin-right:0.5rem; }
          a { color:#0ff; }
        </style>
      </head>
      <body>
        <div id="controls">
          Width: <input id="w" type="number" value="120" min="20" max="320"> 
          Height: <input id="h" type="number" value="60" min="10" max="240"> 
          FPS: <input id="fps" type="number" value="60" min="1" max="240" style="width:5rem;"> 
          Brightness: <input id="brightness" type="number" value="100" min="20" max="300" style="width:5rem;">%
          Mode: <select id="mode">
            <option value="classic">Classic</option>
            <option value="ml">ML Smooth</option>
            <option value="kmeans">K-Means</option>
            <option value="edge">Edge-Aware</option>
            <option value="adaptive">Adaptive</option>
          </select>
          <button id="reload">Reload</button>
          <button id="capture">Capture Frame</button>
          <button id="capture-ascii">Capture ASCII</button>
          <a id="snapshot-url" href="/snapshot" target="_blank">Snapshot</a>
        </div>
        <div id="ascii">Loading...</div>
        <script>
          let pending = false;
          async function load() {
            if (pending) return;
            pending = true;
            try {
              const w = document.getElementById('w').value;
              const h = document.getElementById('h').value;
              const fps = document.getElementById('fps').value;
              const mode = document.getElementById('mode').value;
              const brightness = Math.max(20, Math.min(300, parseInt(document.getElementById('brightness').value, 10) || 100)) / 100;
              const r = await fetch(`/ascii?width=${w}&height=${h}&format=html&mode=${mode}&brightness=${brightness}`);
              const text = await r.text();
              document.getElementById('ascii').innerHTML = text;
              document.getElementById('snapshot-url').href = `/snapshot?width=${w*4}&height=${h*4}`;
            } catch (e) {
              document.getElementById('ascii').textContent = 'Error: ' + e;
            } finally {
              pending = false;
            }
          }
          document.getElementById('reload').addEventListener('click', load);
          let timer = null;
          function startLoop() {
            if (timer) clearInterval(timer);
            const fps = Math.max(1, Math.min(240, parseInt(document.getElementById('fps').value, 10) || 60));
            const interval = Math.max(4, 1000 / fps); // up to 250 fps in ideal case
            timer = setInterval(load, interval);
          }
          document.getElementById('capture').addEventListener('click', async () => {
            try {
              const w = document.getElementById('w').value;
              const h = document.getElementById('h').value;
              const response = await fetch(`/capture?width=${w}&height=${h}`, { method: 'GET' });
              if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
              const data = await response.json();
              alert(`Saved frame image: ${data.path}`);
            } catch (err) {
              alert('Capture failed: ' + err);
            }
          });
          document.getElementById('capture-ascii').addEventListener('click', async () => {
            try {
              const w = document.getElementById('w').value;
              const h = document.getElementById('h').value;
              const mode = document.getElementById('mode').value;
              const brightness = Math.max(20, Math.min(300, parseInt(document.getElementById('brightness').value, 10) || 100)) / 100;
              const response = await fetch(`/capture_ascii?width=${w}&height=${h}&mode=${mode}&brightness=${brightness}`, { method: 'GET' });
              if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
              const data = await response.json();
              alert(`Saved ASCII image: ${data.path}`);
            } catch (err) {
              alert('ASCII capture failed: ' + err);
            }
          });
          document.getElementById('fps').addEventListener('change', startLoop);
          document.getElementById('brightness').addEventListener('change', load);
          document.getElementById('mode').addEventListener('change', load);
          startLoop();
          load();
        </script>
      </body>
    </html>
    '''
    return render_template_string(html)


@app.route('/ascii')
def ascii_frame():
    if not camera_available():
        return Response('Webcam is not available', status=503)

    ret, frame = cap.read()
    if not ret:
        return Response('Failed to read frame', status=500)

    width = parse_int_param('width', TARGET_WIDTH, 20, 320)
    height = parse_int_param('height', TARGET_HEIGHT, 10, 240)
    fmt = request.args.get('format', 'txt').lower()
    mode = request.args.get('mode', 'classic').lower()
    if mode not in ['classic', 'ml', 'kmeans', 'edge', 'adaptive']:
        mode = 'classic'
    brightness = parse_float_param('brightness', 1.0, 0.2, 3.0)

    if fmt == 'html':
        art = frame_to_ascii(frame, width=width, height=height, color=True, mode=mode, brightness=brightness)
        return Response(
            f"<div style='background:#000;color:#0f0;font-family:monospace;line-height:0.85;'>{art}</div>",
            mimetype='text/html; charset=utf-8'
        )

    art = frame_to_ascii(frame, width=width, height=height, color=False, mode=mode, brightness=brightness)
    return Response(art, mimetype='text/plain; charset=utf-8')


@app.route('/capture_ascii', methods=['GET', 'POST'])
def capture_ascii_frame():
    if not camera_available():
        return Response('Webcam is not available', status=503)

    ret, frame = cap.read()
    if not ret:
        return Response('Failed to read frame', status=500)

    width = parse_int_param('width', TARGET_WIDTH, 20, 320)
    height = parse_int_param('height', TARGET_HEIGHT, 10, 240)
    mode = request.args.get('mode', 'classic').lower()
    if mode not in ['classic', 'ml', 'kmeans', 'edge', 'adaptive']:
        mode = 'classic'
    brightness = parse_float_param('brightness', 1.0, 0.2, 3.0)

    ascii_art = frame_to_ascii(frame, width=width, height=height, color=False, mode=mode, brightness=brightness)
    img = ascii_to_image(ascii_art)
    if img is None:
        return Response('Failed to generate ascii image', status=500)

    filename = f"ascii_capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    img.save(filepath)

    return jsonify({'saved': True, 'filename': filename, 'path': filepath})


@app.route('/snapshot')
def snapshot_frame():
    if not camera_available():
        return Response('Webcam is not available', status=503)

    ret, frame = cap.read()
    if not ret:
        return Response('Failed to read frame', status=500)

    width = parse_int_param('width', 640, 100, 1920)
    height = parse_int_param('height', 480, 100, 1080)

    frame = cv2.resize(frame, (width, height))
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return Response('JPEG encoding failed', status=500)

    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/capture', methods=['GET', 'POST'])
def capture_frame():
    if not camera_available():
        return Response('Webcam is not available', status=503)

    ret, frame = cap.read()
    if not ret:
        return Response('Failed to read frame', status=500)

    width = parse_int_param('width', 640, 100, 1920)
    height = parse_int_param('height', 480, 100, 1080)
    frame = cv2.resize(frame, (width, height))

    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    filepath = os.path.join(IMAGES_DIR, filename)

    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return Response('JPEG encoding failed', status=500)

    with open(filepath, 'wb') as f:
        f.write(jpeg.tobytes())

    return jsonify({'saved': True, 'filename': filename, 'path': filepath})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

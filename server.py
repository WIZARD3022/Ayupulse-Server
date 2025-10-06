import time
import numpy as np
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Load .env
load_dotenv()
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

app = Flask(__name__)
CORS(app)

# Buffers for PPG
ppg_buffer = []
timestamps = []

# Dosha score buffers
vata_scores = []
pitta_scores = []
kapha_scores = []
score_timestamps = []

# Table buffer for display
table_buffer = []

# HTML with 3 Chart.js graphs
html_page = """
<!DOCTYPE html>
<html>
<head>
  <title>PPG Dosha Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <meta http-equiv="refresh" content="5">
  <style>
    body { font-family: Arial; margin: 2em; }
    h1 { color: #333; }
    .chart-container {
      width: 100%;
      height: 250px;
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <h1>Live PPG Dosha Dashboard</h1>
  <p><strong>Samples received:</strong> {{ total_samples }}</p>

  <div class="chart-container">
    <canvas id="vataChart"></canvas>
  </div>

  <div class="chart-container">
    <canvas id="pittaChart"></canvas>
  </div>

  <div class="chart-container">
    <canvas id="kaphaChart"></canvas>
  </div>

  <script>
    const times = {{ times|safe }};
    const vataData = {{ vata|safe }};
    const pittaData = {{ pitta|safe }};
    const kaphaData = {{ kapha|safe }};

    function makeChart(id, label, data, color) {
      const ctx = document.getElementById(id);
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: times,
          datasets: [{
            label: label,
            data: data,
            borderColor: color,
            fill: false,
            pointRadius: 0
          }]
        },
        options: {
          animation: false,
          scales: { x: { display: false }, y: { min: 0, max: 1 } }
        }
      });
    }

    makeChart('vataChart', 'Vata Score', vataData, 'blue');
    makeChart('pittaChart', 'Pitta Score', pittaData, 'red');
    makeChart('kaphaChart', 'Kapha Score', kaphaData, 'green');
  </script>

  <table>
    <tr><th>Time</th><th>IR</th><th>Red</th></tr>
    {% for entry in table[::-1] %}
    <tr>
      <td>{{ entry['time'] }}</td>
      <td>{{ entry['ir'] }}</td>
      <td>{{ entry['red'] }}</td>
    </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

@app.route('/data', methods=['POST'])
def receive_data():
    d = request.get_json()
    if d is None:
        return "Invalid", 400

    t = time.time()
    ir = d.get('ir', 0)
    red = d.get('red', 0)

    # Append to buffers
    ppg_buffer.append(ir)
    timestamps.append(t)
    if len(ppg_buffer) > 2000:
        ppg_buffer.pop(0)
        timestamps.pop(0)

    # Table buffer for display
    entry = {'time': time.strftime("%H:%M:%S"), 'ir': ir, 'red': red}
    table_buffer.append(entry)
    table_buffer[:] = table_buffer[-50:]

    # Every 3 seconds → compute dosha scores
    if len(timestamps) > 100 and (t - timestamps[0]) > 3:
        compute_dosha_scores()

    return jsonify({"status": "ok"})

@app.route('/')
def dashboard():
    return render_template_string(
        html_page,
        times=score_timestamps,
        vata=vata_scores,
        pitta=pitta_scores,
        kapha=kapha_scores,
        table=table_buffer,
        total_samples=len(ppg_buffer)
    )

# ==============================
# Simple feature extraction + dosha scoring
# ==============================
def compute_dosha_scores():
    if len(ppg_buffer) < 50:
        return

    ir = np.array(ppg_buffer[-300:], dtype=float)  # last 3 sec @100Hz approx
    ir = ir - np.mean(ir)
    t = np.array(timestamps[-300:], dtype=float)

    # Peak detection
    threshold = np.std(ir) * 0.8
    peaks = [t[i] for i in range(1, len(ir)-1)
             if ir[i] > ir[i-1] and ir[i] > ir[i+1] and ir[i] > threshold]

    if len(peaks) < 2:
        return

    intervals = np.diff(peaks)
    mean_rr = np.mean(intervals)
    hr = 60.0 / mean_rr if mean_rr > 0 else 0
    sdnn = np.std(intervals)
    amp = np.std(ir)

    # --- Heuristic scoring ---
    # Normalize features roughly
    hr_norm = np.clip((hr - 50) / 50, 0, 1)
    var_norm = np.clip(sdnn * 10, 0, 1)
    amp_norm = np.clip(amp / 5000.0, 0, 1)  # adjust depending on sensor scale

    # Vata → high variability, moderate HR
    vata = (var_norm * 0.7 + (1 - abs(hr_norm - 0.5)) * 0.3)

    # Pitta → higher HR, higher amplitude
    pitta = (hr_norm * 0.6 + amp_norm * 0.4)

    # Kapha → lower HR, low variability, low amplitude
    kapha = ((1 - hr_norm) * 0.5 + (1 - var_norm) * 0.25 + (1 - amp_norm) * 0.25)

    # Clamp to 0..1
    vata = float(np.clip(vata, 0, 1))
    pitta = float(np.clip(pitta, 0, 1))
    kapha = float(np.clip(kapha, 0, 1))

    vata_scores.append(vata)
    pitta_scores.append(pitta)
    kapha_scores.append(kapha)
    score_timestamps.append(time.strftime("%H:%M:%S"))

    # Keep buffers limited
    maxlen = 100
    vata_scores[:] = vata_scores[-maxlen:]
    pitta_scores[:] = pitta_scores[-maxlen:]
    kapha_scores[:] = kapha_scores[-maxlen:]
    score_timestamps[:] = score_timestamps[-maxlen:]

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)

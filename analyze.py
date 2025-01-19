from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from werkzeug.utils import secure_filename
import base64
import io
import soundfile as sf
from scipy.signal import butter, lfilter

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def low_pass_filter(data, cutoff, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return img_base64

import soundfile as sf

def analyze_signal(file_path, analysis_type):
    signal, sr = librosa.load(file_path, sr=None)

    # Original Waveform
    fig_original, ax_original = plt.subplots()
    ax_original.plot(np.linspace(0, len(signal) / sr, len(signal)), signal)
    ax_original.set_title("Original Audio Waveform")
    ax_original.set_xlabel("Time (s)")
    ax_original.set_ylabel("Amplitude")
    original_wave_plot = plot_to_base64(fig_original)
    plt.close(fig_original)

    analysis_results = {"original_wave_plot": original_wave_plot}

    if analysis_type == "fft":
        fft = np.fft.fft(signal)
        fft_magnitude = np.abs(fft)
        fft_freq = np.fft.fftfreq(len(fft), 1 / sr)

        fig_fft, ax_fft = plt.subplots()
        ax_fft.plot(fft_freq[:len(fft_freq)//2], fft_magnitude[:len(fft_magnitude)//2])
        ax_fft.set_title("FFT Analysis")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        fft_plot = plot_to_base64(fig_fft)
        plt.close(fig_fft)

        analysis_results["fft_plot"] = fft_plot

    elif analysis_type == "dft":
        dft = np.fft.fft(signal)
        dft_magnitude = np.abs(dft)
        dft_freq = np.fft.fftfreq(len(dft), 1 / sr)

        fig_dft, ax_dft = plt.subplots()
        ax_dft.plot(dft_freq[:len(dft_freq)//2], dft_magnitude[:len(dft_magnitude)//2])
        ax_dft.set_title("DFT Analysis")
        ax_dft.set_xlabel("Frequency (Hz)")
        ax_dft.set_ylabel("Magnitude")
        dft_plot = plot_to_base64(fig_dft)
        plt.close(fig_dft)

        analysis_results["dft_plot"] = dft_plot

    elif analysis_type == "stft":
        stft = librosa.stft(signal)
        stft_magnitude = np.abs(stft)
        stft_time = np.linspace(0, len(signal) / sr, stft_magnitude.shape[1])
        stft_freq = librosa.fft_frequencies(sr=sr)

        fig_stft, ax_stft = plt.subplots()
        ax_stft.imshow(20 * np.log10(stft_magnitude + 1e-6), aspect='auto', origin='lower',
                       extent=[stft_time[0], stft_time[-1], stft_freq[0], stft_freq[-1]])
        ax_stft.set_title("STFT Analysis")
        ax_stft.set_xlabel("Time (s)")
        ax_stft.set_ylabel("Frequency (Hz)")
        stft_plot = plot_to_base64(fig_stft)
        plt.close(fig_stft)

        analysis_results["stft_plot"] = stft_plot

    # Apply low-pass filter
    cutoff_frequency = sr // 4  # Example cutoff frequency
    filtered_signal = low_pass_filter(signal, cutoff=cutoff_frequency, sr=sr)

    # Save the filtered audio
    filtered_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "filtered_" + os.path.basename(file_path))
    sf.write(filtered_audio_path, filtered_signal, sr)

    # Filtered Audio Waveform
    fig_filtered, ax_filtered = plt.subplots()
    ax_filtered.plot(np.linspace(0, len(filtered_signal) / sr, len(filtered_signal)), filtered_signal)
    ax_filtered.set_title("Filtered Audio Waveform")
    ax_filtered.set_xlabel("Time (s)")
    ax_filtered.set_ylabel("Amplitude")
    filtered_wave_plot = plot_to_base64(fig_filtered)
    plt.close(fig_filtered)

    analysis_results["filtered_wave_plot"] = filtered_wave_plot
    analysis_results["filtered_audio_url"] = f"/download/{os.path.basename(filtered_audio_path)}"

    return analysis_results

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'analysis_type' not in request.form:
        return jsonify({"error": "No file or analysis type specified"}), 400

    file = request.files['file']
    analysis_type = request.form['analysis_type']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            analysis_results = analyze_signal(file_path, analysis_type)
            return jsonify(analysis_results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)

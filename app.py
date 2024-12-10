from flask import Flask, render_template, request, jsonify, redirect, url_for
import threading
import queue
import pyaudio
import wave
import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import io
import base64
import time
from scipy.spatial.distance import cosine

class AudioProcessor:
    def __init__(self, audio_file_path, vad_model, classifier, device, speaker_count=2, similarity_threshold=0.85):
        self.audio_file_path = audio_file_path
        self.vad_model = vad_model
        self.classifier = classifier
        self.device = device
        self.speaker_count = speaker_count
        self.similarity_threshold = similarity_threshold

        self.audio_queue = queue.Queue()
        self.speaker_queue = queue.Queue()
        self.waveform_data = []
        self.current_speaker_label = "No speaker detected"
        self.previous_embedding = None

        self.audio_interface = pyaudio.PyAudio()
        self.speaker_index = self.find_usb_speaker_index("Poly Blackwire 3325 Series")

        self.wf = wave.open(audio_file_path, 'rb')
        self.validate_audio_file()

    def find_usb_speaker_index(self, search_name="USB"):
        num_devices = self.audio_interface.get_device_count()
        for i in range(num_devices):
            device_info = self.audio_interface.get_device_info_by_index(i)
            if search_name in device_info.get('name') and device_info.get('maxOutputChannels') > 0:
                print(f"Found USB Speaker: ID = {i}, Name = {device_info.get('name')}")
                return i
        print("USB Speaker not found. Please check your speaker connection.")
        self.audio_interface.terminate()
        exit(1)

    def validate_audio_file(self):
        if self.wf.getnchannels() != 1 or self.wf.getframerate() != 16000:
            print("The audio file must have 1 channel and a sampling rate of 16 kHz.")
            self.wf.close()
            self.audio_interface.terminate()
            exit(1)

    def play_audio(self):
        playback_stream = self.audio_interface.open(
            format=self.audio_interface.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True,
            output_device_index=self.speaker_index
        )

        while True:
            audio_data = self.wf.readframes(8192)
            if len(audio_data) == 0:
                break

            playback_stream.write(audio_data)
            self.audio_queue.put(audio_data)

            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.waveform_data = audio_array.tolist()
            time.sleep(0.1)

        playback_stream.stop_stream()
        playback_stream.close()

    def process_audio(self):
        accumulated_chunk = []
        clustering_model = AgglomerativeClustering(n_clusters=self.speaker_count)

        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break

            accumulated_chunk.append(audio_data)
            if len(accumulated_chunk) < 4:
                continue

            combined_data = b"".join(accumulated_chunk)
            accumulated_chunk = []

            audio_array = np.frombuffer(combined_data, dtype=np.int16).astype(np.float32) / 32768.0
            speech_timestamps = get_speech_timestamps(audio_array, self.vad_model, sampling_rate=16000, threshold=0.6)

            for segment in speech_timestamps:
                start, end = segment['start'], segment['end']
                if end - start < 16000 * 0.5:
                    continue

                if start < len(audio_array) and end <= len(audio_array):
                    speech_chunk = audio_array[start:end]
                    speech_tensor = torch.tensor(speech_chunk, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        embedding = self.classifier.encode_batch(speech_tensor).squeeze().cpu().numpy()

                    if self.previous_embedding is not None:
                        similarity = 1 - cosine(self.previous_embedding, embedding)
                        if similarity < self.similarity_threshold:
                            self.current_speaker_label = "Speaker 2" if self.current_speaker_label == "Speaker 1" else "Speaker 1"
                    else:
                        self.current_speaker_label = "Speaker 1"

                    self.previous_embedding = embedding

                    if self.speaker_queue.empty() or self.speaker_queue.queue[-1] != self.current_speaker_label:
                        self.speaker_queue.put(self.current_speaker_label)

    def generate_waveform_image(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.waveform_data)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Real-Time Waveform')
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return encoded_image

class AudioApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.processor = AudioProcessor(
            audio_file_path="./audio_sample/rahul_mono_long.wav",
            vad_model=vad_model,
            classifier=classifier,
            device=device
        )
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/', methods=['POST'])
        def play():
            threading.Thread(target=self.run_audio_processing).start()
            return redirect(url_for('index'))

        @self.app.route('/waveform', methods=['GET'])
        def get_waveform():
            waveform_image = self.processor.generate_waveform_image()
            return jsonify({
                'waveform_image': waveform_image,
                'speaker_label': self.processor.current_speaker_label
            })

    def run_audio_processing(self):
        play_thread = threading.Thread(target=self.processor.play_audio)
        process_thread = threading.Thread(target=self.processor.process_audio)

        play_thread.start()
        process_thread.start()

        play_thread.join()
        self.processor.audio_queue.put(None)
        process_thread.join()

    def run(self):
        self.app.run(host='0.0.0.0', port=1002, debug=True)

if __name__ == '__main__':
    # Set device to CPU explicitly
    device = torch.device("cpu")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
    get_speech_timestamps, *_ = utils

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb").to(device)
    app = AudioApp()
    app.run()
from online_clustering import OnlineClustering
from utils import VoiceEmbedder

from pyvad import vad
import pyaudio
from collections import deque
import threading
import numpy as np
import time
import torch

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read("config.ini")

RATE = config.getint("AUDIO", "sr")
FORMAT = pyaudio.paFloat32
CHANNELS = 1
FRAME_STRIDE = int(RATE * 0.01)
SECONDS = config.getfloat("AUDIO", "length")

def create_stream():
	pa = pyaudio.PyAudio()
	stream = pa.open(format=FORMAT,
					 channels=CHANNELS,
					 rate=RATE,
					 input=True,
					 output=True,
					 frames_per_buffer=FRAME_STRIDE)
	return stream

stream = create_stream()
buffer = deque(maxlen=int(RATE / FRAME_STRIDE * SECONDS))
def listen():
	while True:
		current_data = stream.read(FRAME_STRIDE, exception_on_overflow = False)
		audio_frame = np.fromstring(current_data, np.float32)
		buffer.append(audio_frame)

# Speaker Embedder model
embedder = VoiceEmbedder()

# online clustering object
cluster_obj = OnlineClustering()

if __name__ == '__main__':

	listen_th = threading.Thread(target=listen)
	listen_th.daemon = True
	listen_th.start()
	times = list() # Profile the prediction inference speed

	try:
		print("Listening...")
		while True:
			if len(buffer) < buffer.maxlen:
				continue
			audio = np.hstack(list(buffer))
			try:
				vact = vad(audio, RATE, fs_vad=RATE, hop_length=30, vad_mode=1)
			except:
				# Skip when audio clips
				continue
			if np.mean(vact) > 0.5:
				audio /= np.max(np.abs(audio))
				audio = audio[None, :]
				now = time.time()
				embeddings = embedder.run(audio)
				prediction = cluster_obj.update_predict(np.squeeze(embeddings))
				times.append(time.time()-now)
				print("                                 ", end="\r")
				print (f"Detected speaker {prediction}", end="\r")
			else:
				pass
				print("                                 ", end="\r")
				print (f"No Speech", end="\r")
			buffer.popleft()
			
	except (KeyboardInterrupt, SystemExit):
		print("\nManual Interrupt")
		if len(times) > 0:
			print(f"Prediction takes ~{np.mean(times)*1000:.2f} ms")

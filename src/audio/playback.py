"""This file handles real-time audio playback using PyAudio."""

import numpy as np
import pyaudio
import threading
import queue
import time
from typing import Optional, Dict, Any
import json
import wave

class AudioPlayback:
    """Handles real-time audio output to speakers"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.playback_queue = queue.Queue()
        self.is_playing = False
        
    def start_playback(self):
        """Start audio playback"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._playback_callback
        )
        self.is_playing = True
        self.stream.start_stream()
        
    def _playback_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio playback"""
        try:
            audio_data = self.playback_queue.get_nowait()
            return (audio_data.tobytes(), pyaudio.paContinue)
        except queue.Empty:
            # Return silence if no audio available
            silence = np.zeros(frame_count, dtype=np.int16)
            return (silence.tobytes(), pyaudio.paContinue)
            
    def play_audio(self, audio_data):
        """Add audio data to playback queue"""
        self.playback_queue.put(audio_data)
        
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

# Voice Conversion Testing Framework

import numpy as np
import pyaudio
import threading
import queue
import time
from typing import Optional, Dict, Any
import json
import wave

# AUDIO CAPTURE & PLAYBACK COMPONENTS

class AudioCapture:
    """Handles real-time audio input from microphone"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def start_capture(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
        
    def get_audio_chunk(self):
        """Get the next audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

class EnhancedAudioCapture(AudioCapture):
    """Audio capture with device selection capabilities"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1, input_device_index=None):
        super().__init__(sample_rate, chunk_size, channels)
        self.input_device_index = input_device_index
        
    def start_capture(self):
        """Start capturing audio from specified device"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,  # Specify device
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()

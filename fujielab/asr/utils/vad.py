"""
utils.vad

Voice Activity Detection utilities using Silero VAD for streaming and chunk-based audio.
"""
import numpy as np
import torch
from typing import List, Optional
from enum import Enum
import pyaudio

class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    This class provides chunk-based VAD using the Silero model.
    Accepts 16bit 16kHz PCM audio chunk (bytes) and detects speech segments.

    Args:
        sampling_rate (int): Sampling rate of the input audio (default: 16000).
        device (str): Device to run the model on (default: 'cpu').
    """
    def __init__(self, sampling_rate: int = 16000, device: str = 'cpu'):
        """
        Initialize SileroVAD.

        Args:
            sampling_rate (int): Sampling rate of the input audio.
            device (str): Device to run the model on.
        """
        self.sampling_rate = sampling_rate
        self.device = device
        self.model, self.utils = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True)
        self.model.to(self.device)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

    def bytes_to_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        """
        Convert 16bit PCM bytes to float32 torch tensor.

        Args:
            audio_bytes (bytes): PCM audio data.

        Returns:
            torch.Tensor: Audio waveform tensor.
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_np).to(self.device)

    def is_speech(self, audio_bytes: bytes) -> bool:
        """
        Determine if the given audio chunk contains speech.

        Args:
            audio_bytes (bytes): PCM audio data.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        wav = self.bytes_to_tensor(audio_bytes)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        # SileroVAD expects at least 100 samples (about 6ms), recommended 0.5s (8000 samples) or more.
        # Here, pad to at least 1536 samples (96ms) as per Silero's recommendation.
        min_samples = 1536
        if len(wav) < min_samples:
            pad = torch.zeros(min_samples - len(wav), dtype=wav.dtype, device=wav.device)
            wav = torch.cat([wav, pad], dim=0)
        with torch.no_grad():
            prob = self.model(wav.unsqueeze(0), self.sampling_rate).item()
        return prob > 0.5

    def get_speech_segments(self, audio_bytes: bytes) -> List[dict]:
        """
        Get speech segment timestamps from the audio chunk.

        Args:
            audio_bytes (bytes): PCM audio data.

        Returns:
            List[dict]: List of speech segment dictionaries with 'start' and 'end' sample indices.
        """
        wav = self.bytes_to_tensor(audio_bytes)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        speech_timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.sampling_rate)
        return speech_timestamps

class VadState(Enum):
    """
    VadState is an enumeration representing the different states of a Voice Activity Detection (VAD) process.

    Attributes:
        NON_SPEECH (int): Indicates that no speech is detected.
        SPEECH_START (int): Indicates the start of a speech segment.
        SPEECH_CONT (int): Indicates that speech is continuing.
        SPEECH_END (int): Indicates the end of a speech segment.
    """
    NON_SPEECH = 0
    SPEECH_START = 1
    SPEECH_CONT = 2
    SPEECH_END = 3

class StreamingSileroVAD:
    """
    Streaming VAD with Silero backend, hop, rollback, and segment output.

    This class provides streaming VAD with configurable hop/frame/trigger parameters.

    Args:
        sampling_rate (int): Sampling rate of the input audio (default: 16000).
        device (str): Device to run the model on (default: 'cpu').
        hop_length_ms (float): Hop size in milliseconds (default: 10.0).
        frame_length_ms (float): Frame size in milliseconds for VAD (default: 100.0).
        rollback_frames (int): Number of frames to roll back at speech start (default: 30).
        speech_trigger_frames (int): Number of consecutive speech frames to trigger speech start (default: 10).
        nonspeech_trigger_frames (int): Number of consecutive non-speech frames to trigger speech end (default: 50).
    """
    def __init__(self, sampling_rate: int = 16000, device: str = 'cpu',
                 hop_length_ms: float = 10.0,
                 frame_length_ms: float = 100.0,
                 rollback_frames: int = 30,
                 speech_trigger_frames: int = 10,
                 nonspeech_trigger_frames: int = 50):
        """
        Initialize StreamingSileroVAD.

        Args:
            sampling_rate (int): Sampling rate of the input audio.
            device (str): Device to run the model on.
            hop_length_ms (float): Hop size in milliseconds.
            frame_length_ms (float): Frame size in milliseconds for VAD.
            rollback_frames (int): Number of frames to roll back at speech start.
            speech_trigger_frames (int): Number of consecutive speech frames to trigger speech start.
            nonspeech_trigger_frames (int): Number of consecutive non-speech frames to trigger speech end.
        """
        self.sampling_rate = sampling_rate
        self.device = device
        self.hop_samples = int((hop_length_ms / 1000.0) * sampling_rate)
        self.hop_bytes = self.hop_samples * 2  # 16bit PCM
        self.frame_samples = int((frame_length_ms / 1000.0) * sampling_rate)
        self.frame_bytes = self.frame_samples * 2  # 16bit PCM
        self.rollback_frames = rollback_frames
        self.speech_trigger = speech_trigger_frames
        self.nonspeech_trigger = nonspeech_trigger_frames
        self.rollback_bytes = self.rollback_frames * self.hop_bytes
        self.vad = SileroVAD(sampling_rate, device)
        self.reset()

    def reset(self):
        """
        Reset the internal state and buffers for streaming VAD.
        """
        self.buffer = bytearray()
        self.frame_results = []
        self.speech_started = False
        self.speech_buffer = bytearray()
        self.speech_count = 0
        self.nonspeech_count = 0
        self.pre_buffer = bytearray()

    def feed(self, audio_bytes: bytes) -> Optional[tuple]:
        """
        Feed PCM bytes. Returns (segment_bytes, VadState) when state changes, else (None, VadState.NON_SPEECH).

        Args:
            audio_bytes (bytes): PCM audio data.

        Returns:
            Optional[tuple]: Tuple of (segment_bytes, VadState). segment_bytes is None if no segment.
        """
        self.buffer.extend(audio_bytes)
        output = None
        state = VadState.NON_SPEECH
        if not hasattr(self, 'returned_length'):
            self.returned_length = 0
        # Maintain a rolling window for frame extraction
        while len(self.buffer) >= self.frame_bytes:
            frame = self.buffer[:self.frame_bytes]
            self.buffer = self.buffer[self.hop_bytes:]
            is_speech = self.vad.is_speech(frame)
            self.frame_results.append(is_speech)
            # pre_bufferはロールバック用にhop_bytesずつ蓄積
            self.pre_buffer.extend(frame[:self.hop_bytes])
            if not self.speech_started:
                if is_speech:
                    self.speech_count += 1
                    self.nonspeech_count = 0
                else:
                    self.speech_count = 0
                if self.speech_count >= self.speech_trigger:
                    # Rollback: go back rollback_frames * hop_bytes
                    rollback = min(len(self.pre_buffer), self.rollback_frames * self.hop_bytes)
                    rollback_data = self.pre_buffer[-rollback:] if rollback > 0 else bytearray()
                    # ロールバック分+現在までの音声全て
                    self.speech_buffer = bytearray()
                    self.speech_buffer.extend(rollback_data)
                    self.speech_buffer.extend(frame[self.hop_bytes:]) # pre_bufferの分は除外
                    self.speech_started = True
                    self.speech_count = 0
                    self.returned_length = 0
                    state = VadState.SPEECH_START
                    output = (bytes(self.speech_buffer), state)
                    self.returned_length = len(self.speech_buffer)
                    break
            else:
                self.speech_buffer.extend(frame[-self.hop_bytes:])
                if not is_speech:
                    self.nonspeech_count += 1
                    self.speech_count = 0
                else:
                    self.nonspeech_count = 0
                if self.nonspeech_count >= self.nonspeech_trigger:
                    # END時は残り全て返す
                    state = VadState.SPEECH_END
                    if self.returned_length < len(self.speech_buffer):
                        segment = self.speech_buffer[self.returned_length:]
                        output = (bytes(segment), state)
                        self.returned_length = len(self.speech_buffer)
                    else:
                        output = (None, state)
                    self.reset()
                    break
                else:
                    # CONT: frame_bytes分溜まってたら未返却分全て返す
                    if self.returned_length + self.frame_bytes <= len(self.speech_buffer):
                        segment = self.speech_buffer[self.returned_length:self.returned_length + self.frame_bytes]
                        state = VadState.SPEECH_CONT
                        output = (bytes(segment), state)
                        self.returned_length += len(segment)
                        break
                    else:
                        output = (None, VadState.SPEECH_CONT)
        if output is not None:
            return output
        else:
            return (None, VadState.NON_SPEECH)

if __name__ == "__main__":
    CHUNK = 320  # 10ms at 16kHz, 16bit mono
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    vad = StreamingSileroVAD(sampling_rate=RATE, device='cpu')

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening... (Ctrl+C to stop)")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            segment, state = vad.feed(data)
            if state == VadState.SPEECH_START:
                print("Speech started")
            elif state == VadState.SPEECH_CONT:
                print("Speech continuing")
            elif state == VadState.SPEECH_END:
                print("Speech ended")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

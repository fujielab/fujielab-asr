import pyaudio
import numpy as np
from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text
from fujielab.asr.utils.vad import StreamingSileroVAD, VadState
import wave

RATE = 16000
CHUNK_MS = 10  # ms
CHUNK = int(RATE * CHUNK_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# モデル名は適宜変更
# model_name = "fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105"
model_name = "fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132"

s2t = Speech2Text.from_pretrained(
    model_name,
    streaming=True,
    lm_weight=0.0,
    beam_size=5,
    beam_search_config=dict(search_type="maes")
)

vad = StreamingSileroVAD(
    sampling_rate=RATE,
    hop_length_ms=CHUNK_MS,
    frame_length_ms=100.0,
    rollback_frames=30,
    speech_trigger_frames=10,
    nonspeech_trigger_frames=100,
)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening... (Ctrl+C to stop)")
try:
    audio_buffer = bytearray()
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        segment, state = vad.feed(data)
        # print("[VAD] State: {}, ({})".format(state.name, len(segment) if segment else "NONE"))

        if segment:
            if state == VadState.SPEECH_START:
                print("[VAD] Speech started")
                audio_buffer = bytearray(segment)
                hyps = s2t.streaming_decode(np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0, is_final=False)
                results = s2t.hypotheses_to_results(hyps)
                if len(results) > 0 and results[0][0]:
                    print(results[0][0])
            elif state == VadState.SPEECH_CONT:
                audio_buffer.extend(segment)
                hyps = s2t.streaming_decode(np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0, is_final=False)
                results = s2t.hypotheses_to_results(hyps)
                if len(results) > 0 and results[0][0]:
                    print(results[0][0])
            elif state == VadState.SPEECH_END:
                print("[VAD] Speech ended. Recognizing (final)...")
                audio_buffer.extend(segment)
                hyps = s2t.streaming_decode(np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0, is_final=True)
                results = s2t.hypotheses_to_results(hyps)
                if len(results) > 0 and results[0][0]:
                    print(results[0][0])
                filename = "output.wav"
                # with wave.open(filename, 'wb') as wf:
                #     wf.setnchannels(CHANNELS)
                #     wf.setsampwidth(p.get_sample_size(FORMAT))
                #     wf.setframerate(RATE)
                #     wf.writeframes(audio_buffer)
                # print(f"Saved audio to {filename}")
                # audio_buffer = bytearray()
except KeyboardInterrupt:
    print("Stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

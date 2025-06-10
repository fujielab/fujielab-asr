import soundfile as sf
import numpy as np
from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text
from scipy.signal import resample
from os import path

# Japanese Pronunciation Combined with Dysfluencies Token Model (ESPnet 202402 required)
model_name = "fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105"
# Japanese Kanji Combined with Dysfluencies Token Model (ESPnet 202402 required)
# model_name = "fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132"
# Japanese Written Character and Special Tokens for Disfluencies Model (ESPnet 202301 required)
# model_name = "fujie/kobori_espnet_0220_transcript-enriched_shojikei_transducer"

s2t = Speech2Text.from_pretrained(
    model_name,
    streaming=True,
    lm_weight=0.0,
    beam_size=20,
    beam_search_config=dict(search_type="maes")
)

# Download a sample audio file if it does not exist.
# The audio file is from https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3
if not path.exists("aps-smp.mp3"):
    import requests
    url = "https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3"
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open("aps-smp.mp3", "wb") as f:
        f.write(response.content)

audio, fs = sf.read("aps-smp.mp3")
if fs != 16000:
    # Resample the audio to 16000 Hz
    num_samples = len(audio)
    audio = resample(audio, int(num_samples * 16000 / fs))
    fs = 16000

num_samples = len(audio)
chunk_size = 16000 * 1/10  # 100ms

final_text = ""
for i in range(0, num_samples, int(chunk_size)):
    chunk = audio[i:i+int(chunk_size)]
    is_final = False
    if len(chunk) < int(chunk_size):
        chunk = np.pad(chunk, (0, int(chunk_size) - len(chunk)), "constant")
        is_final = True
    hyps = s2t.streaming_decode(chunk, is_final=is_final)
    results = s2t.hypotheses_to_results(hyps)
    if len(results) > 0:
        print(results[0][0])
        final_text = results[0][0]

print("Final text:", final_text)

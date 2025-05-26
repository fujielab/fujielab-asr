
import soundfile as sf
import numpy as np
from fujielab_asr.espnet2.bin.asr_transducer_inference_cbs import Speech2Text
from scipy.signal import resample

# Jpaanese Pronounciation Combined with Dysfluencies Token Model
model_name = "fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105"
# Jpaanese Written Character and Special Tokens for Dysfluencies Model
# model_name = "fujie/kobori_espnet_0220_transcript-enriched_shojikei_transducer"

s2t = Speech2Text.from_pretrained(
    model_name,
    streaming=True,
    lm_weight=0.0,
    beam_size=20,
    beam_search_config=dict(search_type="maes")
)

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


"""
エ+F ー+F <sp> パ ラ ゲ ン ゴ ジョ ー ホ ー ト ユ ー | コ ト ナ ン デ ス ガ <sp>
カ ン タ ン ニ | サ イ ショ ニ <sp> エ+F ー+F <sp>
フ ク シュ ー オ | シ テ オ キ タ イ ト | オ モ イ マ ス <sp>
マ+F ア+F ノ+F ー+F | コ ー | ヤ ッ テ | ア+D ッ+D | カ ナ シ テ オ リ マ ス ト |
ソ レ ワ | モ チ ロ ン | ア+F ノ+F | ゲ ン ゴ テ キ ジョ ー ホ ー オ | ツ タ エ ル <sp>
ト ユ ー | コ ト ガ | ヒ ト ツ ノ | ジュ ー ヨ ー ナ | モ ク テ キ ン+D ナ ン デ ア リ マ ス ガ |
 ド ー ジ ニ <sp> パ ラ ゲ ン ゴ ジョ ー ホ ー | ソ シ テ | ヒ ゲ ン ゴ ジョ ー ホ ー <sp>
 ガ | ツ タ ワ ッ テ オ リ マ ス | マ タ | コ ノ | サ ン ブ ン ホ ー ワ | フ ジ サ キ セ ン セ ー
 ニ ヨ ル モ ノ デ シ テ <sp> エ+F ー+F | パ ラ ゲ ン ゴ ジョ ー ホ ー ト ユ ー ノ ワ <sp>
 ヨ ー ワ | ア+F ノ+F | イ ト テ キ ニ | ス ラ イ ド ガ | デ キ ル <sp>
 ワ シャ ガ | チャ ン ト | コ ン ト ロ ー ル シ テ | ダ シ テ ル ン ダ ケ レ ド モ ー <sp>
 ゲ ン ゴ ジョ ー ホ ー ト | チ ガ ッ テ | レ ン ゾ ク テ キ ニ | ヘ ン カ ス ル <sp>
 カ ラ | カ テ ゴ ラ イ ー ズ ス ル | コ ト ガ | ヤ ヤ | ム ツ カ シ ー | ソ ー イ ッ タ |
 ジョ ー ホ ー デ ア リ マ ス <sp>
"""

"""
<dysfl> え ー </dysfl> ぱ ら け ん と し ろ こ う と ゆ う こ と な ん で す が
簡 単 に 最 初 に <dysfl> え ー </dysfl>
九 州 置 き た い と 思 い ま す ー
ま あ <dysfl> あ の ー </dysfl> こ う や っ て <dysfl> あ </dysfl> 悲 し て お り ま す と
そ れ は も ち ろ ん <dysfl> あ の </dysfl> 言 語 を 伝 え る
と ゆ う こ と が 一 つ の 十 四 の 目 的 な ん だ あ り ま す が ー
同 時 に 腹 減 し て 火 ー 言 語 情 報
が 伝 わ っ て お り ま す ま あ こ の 三 分 は 富 士 山 生
に よ る も の で し て ー ぱ ん が 情 報 と ゆ う の は
要 は <dysfl> あ の </dysfl> 伊 藤 的 に 吸 い 方 で き る
わ し わ が ちゃ ん と こ ん と ろ ー る し て 出 し て る ん だ け れ ど も
言 語 帳 ど っ ち が あ っ て ー 連 続 的 に 変 化 す る
か ら 買 っ た こ と な い ー と す る こ と が 嫌 や も ん 難 し い そ う 言 っ た
ら ど こ だ り ま す
"""

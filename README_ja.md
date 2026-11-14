# fujielab-asr

[English](README.md) | 日本語

千葉工業大学 藤江研究室の日本語音声認識です。単語に加えて **フィラー・言い直し** も書き起こします。

`fujielab-asr` は、いくつかの種類の認識器を同じ `from_pretrained` の呼び方で使えるようにしたパッケージです:
音素列と書字形を出す 2 層の Zipformer（ストリーミング版とオフライン版。icefall で学習し、TorchScript で実行）、
sherpa-onnx で動かすストリーミング Zipformer、[ESPnet](https://github.com/espnet/espnet) の上に作った
Contextual Block Streaming（CBS）エンコーダのストリーミング Transducer / CTC（`fujielab.asr.espnet_ext`）、
Whisper large-v3 + LoRA。ストリーミングのモデルは音声を少しずつ（たとえば 100 ms ごとに）受け取り、
ほとんどのモデルは CPU で実時間で動きます。

## モデルの入手

この README に載っている学習済みモデルは、Hugging Face Hub 上で **非公開（private）** です。
パッケージのコードは Apache License 2.0 で公開しています。
モデルの利用を希望される方は、藤江（ **shinya.fujie@chibatech.ac.jp** ）までご連絡ください。
ただし、学習に使ったコーパスの条件により、研究室の外へのモデルの提供には制限があります（下の「ライセンス」を参照）。

ここに載っている例は、研究室内部での利用を想定しています。
`from_pretrained` は、モデルを読めるアカウントでログイン
していれば（`hf auth login` か `HF_TOKEN`）Hub からモデルをダウンロードします。
手元にあるモデルも読み込めます（ESPnet と Whisper のモデルはまとめた `.zip`、sherpa-onnx と Cascade の
モデルはモデルのフォルダを渡します）。

## 特長

認識器（下の「学習済みモデル」と同じ順）:

- **2 層・2 レートのストリーミング Zipformer（Cascade）** （`Speech2TextCascade`）— 1 つの音声から、
  フィラー・言い直し・笑いの印付きの **音素列** を 80 ms ごとに（層 1）、 **書字形** （漢字かな交じり）を
  160 ms ごとに（層 2）、先読みなしで出します。icefall で学習し、TorchScript として CPU で実時間で動きます。
  ReazonSpeech（テレビ放送）の疑似ラベルを学習データに加えた低遅延モデルで、CSJ+CEJC の評価セットにおいて
  層 1 の P-TER 7.80 %（単語の遅れ 209 ms）、層 2 の CER 11.53 %（345 ms）。
  `cascade` extra が必要です: `pip install "fujielab-asr[cascade]"`。
- **2 層オフライン Zipformer** （`Speech2TextCascadeOffline`）— このパッケージでいちばん精度の高いモデルで、
  録音した音声の書き起こし向けです。両層が発話の全体を見て、 **書字形そのものにフィラー・言い直しの印が付きます**
  （`(F えー)昨日(D こ)今度の話`、CSJ の表記）。層 2 の CER 9.37 %、書字形でのフィラー / 言い直しの F1 84.6 / 59.7、
  層 1 の P-TER 6.30 %。CPU 4 スレッドで実時間の 0.03 倍。大きい版（層 1 が Zipformer large、1.9 億パラメータ）は
  層 2 の CER が 9.05 %、実時間の 0.04 倍です。同じ `cascade` extra を使います。
- **sherpa-onnx によるストリーミング Zipformer Transducer** （`Speech2TextSherpaOnnx`）— icefall で学習した
  6,600 万パラメータのモデルを ONNX に書き出し、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) で動かします。
  **CPU 1 スレッドで実時間** （320 ms チャンクで RTF 0.08 fp32 / 0.05 int8）、チャンクごとに途中結果、
  認識時に PyTorch 不要。CSJ+CEJC の評価セットで CER 12.24 %（CBS RNN-T: 17.02 %）。
  `sherpa` extra が必要です: `pip install "fujielab-asr[sherpa]"`。
- **Transducer マルチタスク** （`Speech2TextMultitaskTransducer`）— CBS Conformer エンコーダ（ESPnet）による
  ストリーミングの Transducer ビーム探索に加えて、
  別のヘッドでトークンごとに **補助情報のラベル** を予測します: `N`（通常）、`F`（フィラー）、`D`（言い直し）。
- **RNN-Transducer** （`Speech2Text`）— 補助ヘッドの無い、同じストリーミングのビーム探索。
- **CTC マルチタスク** （`Speech2TextMultitaskCTC`）— 同じ補助ラベルを、ストリーミングの貪欲 CTC 認識器で出します。
- **CTC** （`Speech2TextCTC`）— 補助情報の印が認識トークンに含まれるモデル（合成トークン `あ+F` / `チ+D`、
  または区間タグ `<F> あ の </F>`）向けの、ストリーミングの貪欲 CTC。
- **Whisper large-v3 + LoRA** （`Speech2TextWhisper`）— Whisper を LoRA で同じモーラ＋印の出力
  （`token_type whisper_mora`）に適応させたもの。ストリーミング向けに **ブロック因果エンコーダ**
  （100 ms チャンク、先読みの上限付き）も選べます。今は発話単位の認識で、チャンクごとの API は予定しています。
  `whisper` extra が必要です: `pip install "fujielab-asr[whisper]"`。

Transducer の探索は `search_type="maes"` で使います（ここの例はすべてそうしています）。
`BeamSearchTransducer` の `default` / `tsd` / `alsd` は、このパッケージのストリーミング用 API より前のもので、
今は動きません。

## インストール

### 必要なもの

- **Python 3.10 〜 3.12** （3.11 で確認）。
- **ESPnet 202412 〜 202511** — 依存パッケージとして自動で入ります。
  （古い `fujielab-asr`（0.1.3 以前）は ESPnet 202301〜202503 向けでした。0.1.4 からは `typeguard` 4.x を使う
  新しい ESPnet に合わせています。ESPnet 202609 以降には、この拡張が使う `espnet` パッケージが含まれていません。）
- `torch` / `torchaudio`（CUDA や環境に合ったものを入れてください）。torchaudio は、対応する torch とでないと
  読み込めません。torchaudio 2.11 はその対応を正しく宣言していない（2.11.1 は torch 2.11・2.12 と組み合わせると
  読み込めない）ので、このパッケージは両方を 2.11 未満に制限しています。2.11 未満では torchaudio がそれぞれ
  対応する torch を固定しているので、pip は torch 2.10.0 + torchaudio 2.10.0 を入れます。版を自分で選ぶときは、
  `pip install fujielab-asr "torch==2.7.1" "torchaudio==2.7.1"` のように両方を指定してください。
  読み込めない組み合わせで ESPnet の認識器を import すると、その旨の `ImportError` が出ます。

ESPnet の依存パッケージにはソースからビルドするもの（`pyworld` など）があるので、C/C++ のコンパイラが
使える環境をおすすめします。

### PyPI からインストール

```bash
pip install fujielab-asr
```

新しい仮想環境に入れるのがおすすめです:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install fujielab-asr
```

### ソースからインストール

```bash
git clone https://github.com/fujielab/fujielab-asr
cd fujielab-asr
pip install -e .
```

## 学習済みモデル

`from_pretrained` にタグを渡して読み込みます。すべて非公開です（「モデルの入手」を参照）。
下の「使い方」と同じ順に並べています。

1. **2 層ストリーミング Zipformer（Cascade）** — `Speech2TextCascade`: フィラー・言い直し・笑いの印付きの音素列と
   書字形。CPU で実時間（`cascade` extra。[使い方](#1-2-層ストリーミング-zipformercascade)）
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_stream_c80_c160_la0`
     — **おすすめ** : 低遅延（80 / 160 ms チャンク、先読みなし）、CSJ+CEJC+ReazonSpeech
   - `fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
     — 160 / 640 ms チャンク、計算量は約半分、CSJ+CEJC+ReazonSpeech
   - `fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
     — CSJ+CEJC のみ。`…_fold1` / `…_fold2` は fold 1 / 2 の話者を除いて学習したもの
   - `fujielab/icefall_asr_csj_cejc_phn_writ_comp_cascade_zipformer_stream_c160_c640_la0` — 以前の版（イベントなし）
2. **2 層オフライン Zipformer** — `Speech2TextCascadeOffline`: 録音の書き起こし向け。書字形そのものに
   `(F えー)`・`(D こ)` の印が付きます（`cascade` extra。[使い方](#2-2-層オフライン-zipformer)）
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond` — **既定**
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond`
     — 大きい版（1.9 億パラメータ）: 少し精度が高く、時間は約 1.4 倍
3. **sherpa-onnx によるストリーミング Zipformer** — `Speech2TextSherpaOnnx`: 合成トークンの印付きのカナ。
   CPU 1 スレッドで実時間、PyTorch 不要（`sherpa` extra。[使い方](#3-sherpa-onnx-によるストリーミング-zipformercpu-で実時間)）
   - `fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0`
4. **ストリーミング Transducer（ESPnet、CBS）** — カナ、CSJ+CEJC（[使い方](#4-ストリーミング-transducer-マルチタスク認識補助情報)）
   - `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132`
     — `Speech2TextMultitaskTransducer`: 別の `N`/`F`/`D` ヘッド
   - `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_transducer_120300_hop132`
     — `Speech2Text`: 合成トークン `あ+F` / `チ+D`
   - `fujielab/espnet_asr_csj_cejc_pron_span_cbs_transducer_120300_hop132`
     — `Speech2Text`: 区間タグ `<F> あ </F>`
5. **ストリーミング CTC（ESPnet、CBS）** — カナ（[使い方](#5-ストリーミング-ctc-マルチタスク認識補助情報)）
   - `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_ctc_120300_hop132` — `Speech2TextMultitaskCTC`
   - `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_ctc_120300_hop132`、
     `fujielab/espnet_asr_csj_cejc_pron_span_cbs_ctc_120300_hop132` — `Speech2TextCTC`
   - 同じ 3 種類を CSJ のみで学習したもの: `fujielab/espnet_asr_csj_pron_{aux,comp,span}_cbs_ctc_120300_hop132`
6. **Whisper large-v3 + LoRA** — `Speech2TextWhisper`: 合成トークンの印付きのカナ、ブロック因果エンコーダ
   （100 ms チャンク）、GPU（`whisper` extra。[使い方](#6-whisper-のモデル)）
   - `fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0`
7. **以前のストリーミング Transducer（ESPnet、CBS）** — `Speech2Text`（[使い方](#7-以前のストリーミング-transducer)）
   - `fujielab/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132` — CEJC、カナ
   - `fujielab/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132` — CSJ、漢字
   - `fujielab/espnet_asr_cbs_transducer_120303_hop132_cc0105` — CEJC+CSJ、カナ

## 使い方

上と同じ順に、モデルの種類ごとに、インストール・最小の例・`examples/` の実行できるスクリプト・結果を示します。
入力には `examples/aps-smp.mp3`（CSJ のサンプル音声、35 秒）が使えます。

### 1. 2 層ストリーミング Zipformer（Cascade）

`Speech2TextCascade` は、pron-based-asr プロジェクトの 2 層・2 レートの認識器を動かします。
[icefall](https://github.com/k2-fsa/icefall) で CSJ + CEJC（おすすめのモデルは疑似ラベル付きの ReazonSpeech も）
を学習し、1 つの TorchScript モジュール（8,450 万パラメータ、345 MB）に書き出したものです。
1 つの音声から、

- **層 1** （fbank を入力とする因果的 Zipformer Transducer。低遅延モデルは 80 ms チャンク、ほかは 160 ms）が、
  IPA にもとづく表記の **音素** （`ː` 長音、`N` 撥音、`Q` 促音、`ʲ` 拗音、`|` 単語の区切り、`<sp>` ポーズ）を、
  イベントの印付きで出します: `a+F` はフィラー中の音素、`a+D` は言い直し・語断片中の音素、Event-DP モデルでは
  `<laugh>`（笑い）と `<L>` … `</L>`（笑いながらの発話）も出ます。音声対話での話者交替や相づちの判断に
  使える速さです。
- **層 2** （層 1 のエンコーダ出力を入力とする小さめの因果的 Zipformer。低遅延モデルは 160 ms チャンク、
  ほかは 640 ms）が、 **書字形** （漢字かな交じり）を、層 1 より約 0.15 秒遅れて（640 ms チャンクでは 0.3〜0.6 秒）出します。

必要なのは `torch`、`kaldi-native-fbank`、`numpy`、`huggingface_hub` だけで、ESPnet と icefall は import しません。

```bash
pip install "fujielab-asr[cascade]"
```

```python
import soundfile as sf
from fujielab.asr.icefall_ext import Speech2TextCascade

s2t = Speech2TextCascade.from_pretrained(
    "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_stream_c80_c160_la0",
    num_threads=4,
)

audio, fs = sf.read("utterance.wav", dtype="float32")   # 16 kHz mono
step, n, result = int(fs * 0.1), len(audio), None        # 100 ms per call
for i in range(0, n, step):
    result = s2t.streaming_decode(audio[i:i + step], is_final=i + step >= n)[0]
    # result.kana / result.text are the hypotheses so far
print(result.kana)    # "エ、パラゲンゴジョーホート ユー コトナ ンデスガ ア カンタンニ ..."  (layer 1)
print(result.text)    # "えパラ言語情報ということなんですがあ簡単に ..."               (layer 2)
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.base_tokens, result.aux_labels)))   # "e[F] <sp> p a r a ..."
```

`result.tokens` / `aux_labels` / `base_tokens` / `timestamps` / `confidences` は、1 対 1 に対応した層 1 の記号です
（このパッケージのほかの認識器と同じ）。`result.graphemes` / `grapheme_timestamps` が層 2 です。少しずつ使うときは、
`s2t.accept(chunk)` と `s2t.flush()` が各層の新しい記号だけを `(フレーム, 記号, 確信度)`（40 ms フレーム）で返します。
次の発話の前に `s2t.reset()` を呼んでください。読み込むと、プロセス全体で TorchScript の profiling executor を
止めます（止めないと最初の decoder の呼び出しが失敗します）。eager モードの ESPnet モデルには影響しません。
Cascade のモデルはどれも同じ呼び方・同じ出力記号で（イベントなしの以前のモデルに笑いの記号が無いだけ）、
チャンクの大きさは各モデルの `config.json` から読むので、入れ替えて使えます。低遅延モデルの計算量は
160 / 640 ms のモデルの約 2 倍です。タグを渡さない `from_pretrained()` は、CSJ+CEJC のみの Cascade+Event-DP モデル
（`fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`）を読み込みます。

**ライブデモ.** `examples/run_cascade_live.py` は、話している最中に層 1（カタカナ）と層 2 を並べて表示します
（ノート PC の CPU で動きます）:

```bash
pip install "fujielab-asr[cascade]" sounddevice soxr rich
python examples/run_cascade_live.py                         # microphone
python examples/run_cascade_live.py --wav examples/aps-smp.mp3   # a file, in real time (--fast: full speed)
python examples/run_cascade_live.py --bench                 # real-time factor, per layer
```

デモは既定で低遅延モデルを読み込みます。処理が追いつかないと状態表示に出たら、
`--model fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
で同じ精度のまま計算量が約半分になります。`--model` にはほかのタグや手元のモデルのフォルダも渡せます。

CSJ + CEJC の評価セット 17 個（23,259 発話、貪欲探索）での結果です。P-TER は層 1 の記号すべてを数え、CER は書字形で
数えています。単語の遅れ（中央値）は、実際のストリーミングで単語の最後の記号を出せる時刻から、CSJ の時刻情報での
単語の終わりを引いたものです（CSJ のコア講演 8 個）:

| モデル | チャンク（層 1 / 層 2） | 層 1 P-TER % | 層 1 の遅れ | 層 2 CER % | 層 2 の遅れ | 笑い F1 | フィラー F1 | 言い直し F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 低遅延 Cascade+Event-DP + ReazonSpeech（`…_reazon_large_…_c80_c160_la0`） | 80 / 160 ms | **7.80** | **209 ms** | **11.53** | **345 ms** | 80.8 | 74.1 | 48.7 |
| Cascade+Event-DP + ReazonSpeech（`…_reazon_medium_…_c160_c640_la0`） | 160 / 640 ms | 7.81 | 279 ms | 11.55 | 583 ms | **81.6** | **74.4** | **49.5** |
| Cascade+Event-DP、CSJ+CEJC のみ（`…_csj_cejc_phn_…_ev_cascade_…`） | 160 / 640 ms | 8.05 | 275 ms | 12.30 | 562 ms | 80.5 | 72.5 | 46.3 |
| Cascade（`…_cascade_…`） | 160 / 640 ms | 7.97 | 356 ms | 12.31 | 589 ms | — | — | — |
| 1 層の書字 Zipformer | 160 ms | — | — | 13.53 | | — | — | — |

160 / 640 ms のモデルは各チャンクの先 130 ms も読みますが、低遅延モデルは読みません。低遅延モデルは、
160 / 640 ms の ReazonSpeech モデルと同じ精度で（層 2 CER −0.03 [−0.16, +0.10]、P-TER −0.01 [−0.10, +0.07]、
対応のあるブートストラップ）、単語が 70 ms（層 1）・238 ms（層 2）早く出ます。ReazonSpeech を加えると、
CSJ+CEJC で層 2 の CER が 0.74 ポイント下がり（[−0.88, −0.63]）、それ以外のデータではもっと大きく下がります（層 2 CER %）:

| モデル | JSUT basic5000（朗読） | TEDxJP-10K（講演） | LaboroTV dev（テレビ） | ReazonSpeech test（テレビ） |
|---|---:|---:|---:|---:|
| 低遅延 Cascade+Event-DP + ReazonSpeech（large） | **9.54** | 13.79 | **15.44** | **9.71** |
| Cascade+Event-DP + ReazonSpeech（medium） | 10.01 | **13.31** | 15.77 | 10.34 |
| Cascade+Event-DP、CSJ+CEJC のみ | 12.00 | 16.04 | 23.38 | 22.14 |

JSUT は数字を漢字にして（CSJ / CEJC と同じ）採点しています。ReazonSpeech test は学習に使っていない 2,000 発話で、
字幕を正解とし、認識結果からフィラーを除いて採点しています。ReazonSpeech は株式会社レアゾン・ホールディングスが
独自の条件（CDLA-Sharing-1.0、著作権法第 30 条の 4 の範囲での利用）で配布しています。それが問題になる場合は、
CSJ+CEJC のみのモデルを使ってください。

ストリーミングの実装は評価時の認識を再現しています（Cascade+Event-DP、150 発話: P-TER 6.36 対 6.35、CER 7.66 対 7.74。
違いはほとんどが発話の最後の 1〜2 文字です）。i9-12900K で 4 スレッドなら RTF 0.12（1 スレッドで 0.16）、
低遅延モデルは 0.22（0.27）です。学習の詳細は各モデルカードをご覧ください。

### 2. 2 層オフライン Zipformer

`Speech2TextCascadeOffline` は、同じプロジェクトのオフラインモデル
（`fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond`、
8,410 万パラメータ、344 MB）を動かします。リアルタイムの音声ではなく、録音した音声の書き起こし向けです。
両層が **発話の全体** を見ます。これが最も効くのは言い直しです（語が言いさしになったかどうかは、その後を聞かないと
わかりません）。ストリーミングのモデルと比べて、

- **層 2 自身がフィラー・言い直しの印を出す** （`え+F`、`こ+D`）ので、書字形が CSJ の表記
  `(F えー)昨日(D こ)今度の話` で出ます。笑いは `<笑>` / `(笑 …)` です。
- **層 2 は層 1 の判断も読みます** : 層 1 の貪欲探索が各フレームで出した音素・印の記号を、層 1 のエンコーダ出力と
  あわせて層 2 のエンコーダに入れます。

```python
import soundfile as sf
from fujielab.asr.icefall_ext import Speech2TextCascadeOffline

s2t = Speech2TextCascadeOffline.from_pretrained(num_threads=4)   # the model above
audio, fs = sf.read("utterance.wav", dtype="float32")             # 16 kHz mono, one utterance
r = s2t(audio)[0]
print(r.tagged_text)  # "(F まあのー)こうやって(D あ)話しておりますとそれはもちろん(F あの)言語的情報を ..."
print(r.text)         # "まあのーこうやってあ話しておりますとそれはもちろんあの言語的情報を ..."
print(r.kana)         # "マアノー コー ヤッテ ハ ハナシテ オリマスト ..."   (layer 1)
```

結果はストリーミングのモデルと同じ `CascadeResult` です。`base_graphemes` / `grapheme_aux_labels` は、層 1 での
`base_tokens` / `aux_labels` と同じように、印付きの層 2 の文字を文字と印に分けたものです。時刻は発話の先頭からの
秒数です。`s2t.decode_fbank(feats)` には、音声の代わりに計算済みの 80 次元 fbank（lhotse の特徴量など）を渡せます。
**長い録音は先にポーズで区切ってください** : 学習に使った発話は最長 30 秒（99 % は 17.7 秒以内）です。
サンプルスクリプトは、簡単なパワーにもとづく区切りでこれを行います:

```bash
pip install "fujielab-asr[cascade]"
python examples/run_cascade_offline.py examples/aps-smp.mp3              # per segment: layer 1 and layer 2
python examples/run_cascade_offline.py a.wav --out result.jsonl          # also every symbol with its time
```

**大きい版.** `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond`
は、ファイル・呼び方・探索が同じで、層 1 が Zipformer large です（エンコーダの出力が 512 / 384 次元ではなく
768 / 512 次元。1 億 9,000 万パラメータ、768 MB）。どの評価でも少し精度が高く（下の表）、時間は約 1.4 倍かかります。
タグを `from_pretrained` に（スクリプトでは `--model` に）渡します:

```python
s2t = Speech2TextCascadeOffline.from_pretrained(
    "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond", num_threads=4)
```

CSJ + CEJC の評価セット 17 個（貪欲探索）での結果です。ここでのフィラー / 言い直しの F1 は **書字形** で数えています
（編集距離で文字を対応付けて印を比べる）。上のストリーミングの表の層 1 のイベント F1 とは違います。層 2 に印の無い
モデルは、層 1 の印を時刻の対応で移して数えています:

| モデル | 層 2 CER % | 印込みの CER % | フィラー F1 | 言い直し F1 | 層 1 P-TER % | JSUT / TEDxJP / LaboroTV（CER %） |
|---|---:|---:|---:|---:|---:|---|
| **オフライン Cascade、印付きの層 2 + 層 1 の入力** （既定のモデル） | 9.37 | 10.02 | **84.6** | 59.7 | 6.30 | 7.90 / 10.37 / 13.71 |
| オフライン Cascade large（`…_zipformer_l_offline_writaux_proncond`）、印付きの層 2 + 層 1 の入力 | **9.05** | **9.70** | **84.6** | **59.8** | **6.13** | **7.50** / **10.32** / **13.34** |
| オフライン Cascade、印なしの層 2、同じデータ・同じ学習（非公開） | 9.33 | 10.49 | 81.8 | 54.0 | 6.36 | 8.04 / 10.42 / 13.61 |
| ストリーミングの低遅延 Cascade（`…_c80_c160_la0`） | 11.53 | 12.91 | 79.7 | 44.1 | 7.80 | 9.54 / 13.79 / 15.44 |

層 2 に印を付けても CER は悪くならず（印なしの層 2 に対して +0.04、有意差なし）、フィラー / 言い直しの F1 は
2.8 / 5.7 ポイント上がります。実装は評価時の認識を再現しています: 評価の 10 発話で、同じ fbank からなら両層の記号と
フレーム時刻が icefall と一致し、音声から（kaldi-native-fbank）でも記号が一致します（torch 2.6、2.7、2.10、2.12 で
確認。大きい版も torch 2.10 で同じ確認をしています）。i9-12900K で 4 スレッドなら RTF 0.028（1 スレッドで 0.052）、
大きい版は 0.041（0.079）です。学習の詳細はモデルカードをご覧ください。

### 3. sherpa-onnx によるストリーミング Zipformer（CPU で実時間）

`Speech2TextSherpaOnnx` は、icefall で学習したストリーミング Zipformer Transducer
（`fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0`）を
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) で動かします。ダウンロードするのは指定したチャンクの大きさ・
精度の ONNX ファイルだけです（fp32 のエンコーダ 261 MB、int8 70 MB）。`torch` と ESPnet は import しません。
出力トークンは CBS の `comp` モデルと同じ合成トークン（`ア+F`、`ン+D`、`|`、`<sp>`、`<mask>`）で、結果には
マルチタスクの認識器と同じく 1 対 1 に対応した `tokens` / `aux_labels` / `base_tokens` があります。

```bash
pip install "fujielab-asr[sherpa]"
```

```python
import soundfile as sf
from fujielab.asr.sherpa_onnx_ext import Speech2TextSherpaOnnx

s2t = Speech2TextSherpaOnnx.from_pretrained(
    "fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0",
    chunk_ms=320,          # 320 (default) or 640
    int8=False,            # True: int8 encoder, ~1.8x faster
    num_threads=1,
)

audio, fs = sf.read("utterance.wav", dtype="float32")   # 16 kHz mono
step, n, result = int(fs * 0.1), len(audio), None        # 100 ms per call
for i in range(0, n, step):
    result = s2t.streaming_decode(audio[i:i + step], is_final=i + step >= n)[0]
    # result.text is the partial hypothesis so far
print(result.text)        # "ア+F ノ+F ー+F | ス+D | イ ッ パ ツ メ カ ラ ー"
print(result.transcript)  # "アノースイッパツメカラー"
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.base_tokens, result.aux_labels)))
```

`is_final=True` は 0.6 秒の無音を足して最後のトークンを出し切ります（モデルは末尾に無音のある音声で学習しているので、
発話の終わりに余計なトークンが出るのもこれで抑えられます）。次の呼び出しから新しい発話になります。
発話の終わりを自動で検出して使うときは、`enable_endpoint_detection=True` を渡し、`s2t.is_endpoint()` /
`s2t.finish_segment()` を使ってください。

CSJ+CEJC の評価セット 17 個での結果です（CER は `+F`/`+D` の印を含めてトークンすべてを数え、`<sp>`/`<mask>` は
1 記号として数える。RTF は CPU 1 スレッド、100 ms ずつ入力）:

| モデル | チャンク | パラメータ | CER % | F F1 | D F1 | RTF |
|---|---|---:|---:|---:|---:|---:|
| Zipformer、貪欲探索 | 320 ms | 66 M | **12.24** | 84.7 | 54.4 | 0.084（int8 0.046） |
| Zipformer、modified beam search 4 | 320 ms | 66 M | 11.82 | 85.1 | 55.3 | |
| Zipformer、貪欲探索 | 640 ms | 66 M | 12.37 | 84.8 | 55.2 | 0.051（int8 0.029） |
| CBS Conformer RNN-T `comp`（このパッケージ、ビーム 20） | 100 ms | 30 M | 17.02 | 82.0 | 45.5 | 0.275 |
| Whisper large-v3 + LoRA、ブロック因果、先読み 0 | 100 ms | 1.55 B | 12.25 | 85.3 | 57.6 | GPU |

CBS RNN-T と比べて Zipformer は CER が 4.8 ポイント良く（対応のあるブートストラップで [−4.94, −4.64]）、3.3 倍速く、
ストリーミングの Whisper とは CER が同等で（−0.01、有意差なし）、パラメータは 1/23 です。

スクリプト: `examples/run_streaming_asr_sherpa.py`（100 ms ごとに途中結果）。

### 4. ストリーミング Transducer マルチタスク（認識＋補助情報）

```python
import numpy as np, soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_transducer_inference_cbs import (
    Speech2TextMultitaskTransducer,
)

s2t = Speech2TextMultitaskTransducer.from_pretrained(
    "fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132",
    beam_size=20, beam_search_config=dict(search_type="maes"),
)

audio, fs = sf.read("utterance.wav")   # 16 kHz mono
chunk = int(16000 * 0.1)               # 100 ms
n, result = len(audio), None
for i in range(0, n, chunk):
    c = audio[i:i + chunk]
    is_final = i + chunk >= n          # flag the last chunk even when it is full
    if len(c) < chunk:
        c = np.pad(c, (0, chunk - len(c)))
    r = s2t.streaming_decode(c, is_final=is_final)
    if r:
        result = r[0]

# result.tokens and result.aux_labels are aligned 1:1 (aux in {N, F, D})
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.tokens, result.aux_labels)))
```

補助ヘッドは joint network の隠れ状態の上にあるので、ラベルは格子の特定のノード（フレームと予測器の状態の組）に
しか存在しません。研究用のオフラインのレシピは、発話を符号化し直して最終仮説を強制アライメントすることで
そのノードを求めます（完全な音声に対する 2 回目の処理）。このパッケージは代わりに、ビーム探索がトークンを出した
ノードでヘッドを読むので、2 回目の処理が要らず、チャンクごとに補助ラベルが得られます。CEJC の 297 発話（学習に
不使用）で、2 つの方法はトークンの **99.5 %** （どちらかが `F` か `D` と言うトークンでは **90.2 %** ）で一致します。

`comp` / `span` の Transducer モデルは印をトークンに含めていて、素の `Speech2Text`
（`fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs`）を同じ 100 ms ずつのループで使います
（使い方は 7 を参照）。スクリプト: `examples/run_streaming_asr_multitask_transducer.py`（`aux`）と
`examples/bench_streaming_rtf.py`（CPU での実時間比）。

### 5. ストリーミング CTC マルチタスク（認識＋補助情報）

```python
import numpy as np, soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_ctc_inference_cbs import (
    Speech2TextMultitaskCTC,
)

s2t = Speech2TextMultitaskCTC.from_pretrained(
    "fujielab/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132", streaming=True
)

audio, fs = sf.read("utterance.wav")  # 16 kHz mono
chunk = int(16000 * 0.1)              # 100 ms
for i in range(0, len(audio), chunk):
    c = audio[i:i + chunk]
    is_final = len(c) < chunk
    if is_final:
        c = np.pad(c, (0, chunk - len(c)))
    r = s2t.streaming_decode(c, is_final=is_final)[0]
    # r.tokens and r.aux_labels are aligned 1:1 (aux in {N, F, D})
    print(" ".join(f"{t}[{a}]" if a != "N" else t
                   for t, a in zip(r.tokens, r.aux_labels)))
```

スクリプト: `examples/run_streaming_asr_multitask.py`（`aux`、`Speech2TextMultitaskCTC`）と
`examples/run_streaming_asr_ctc.py`（`comp` / `span`、`Speech2TextCTC`）。

**ESPnet のモデルの選び方（4 と 5）.** CSJ+CEJC の ESPnet の 6 モデル（印の表し方 3 種類 × ヘッド 2 種類）は、
エンコーダ・データ・学習スケジュールが共通です。CER は、正解・認識結果の両方から印をすべて除いて数えています:

| 印の表し方 | RNN-T CER % | CTC CER % | RNN-T F/D F1 | CTC F/D F1 |
|---|---:|---:|---:|---:|
| `aux` | **16.01** | 16.90 | 81.9 / 44.4 | 81.6 / 43.1 |
| `comp` | 16.25 | 17.64 | **82.0** / **45.5** | 80.4 / 44.3 |
| `span` | 16.41 | **16.74** | 81.5 / 43.9 | 81.3 / 41.6 |

- **ヘッド** : どの表し方でも RNN-T が CTC より良い（+0.33〜+1.39 ポイント、23,500 発話ですべて有意）。
  小さく、decoder の無いモデルが欲しければ CTC を選んでください。
- **印の表し方** : 差は学習ごとのばらつきの範囲内で、順位を読み取らないでください。フィラー / 言い直しの F1 も、
  どちらを選んでもほとんど変わりません（`D` の再現率はどれも 34〜38 %）。
- 区間タグは開閉が揃わずに出ることがあります（CTC 11.1 %、RNN-T 2.8 % の発話）。`<F>`/`<D>` は「ここからラベル」、
  `</F>`/`</D>` は「ラベル終わり」として読んでください。
- CSJ のみの CTC の 3 モデル: 内容の CER ≈6.7〜6.8 %、フィラー F1 ≈93。

### 6. Whisper のモデル

`Speech2TextWhisper` は Whisper large-v3 + LoRA のモデルを読み込みます。Hub からダウンロードするのはアダプタ
（LoRA の行列と追加トークンの埋め込み、約 70 MB）だけで、Whisper の本体の重み（3 GB）は初回に `openai-whisper` が
取得します。

```bash
pip install "fujielab-asr[whisper]"
python examples/run_whisper_asr.py utterance.wav --device cuda
```

```python
import soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_whisper_inference import Speech2TextWhisper

s2t = Speech2TextWhisper.from_pretrained(
    "fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0", device="cuda")
audio, fs = sf.read("utterance.wav")      # 16 kHz mono, <= 30 s
print(s2t(audio)[0].text)                 # "ア+F ノ+F ー+F | ス+D | イ ッ パ ツ メ ..."
```

CSJ+CEJC の評価セット 17 個で、ブロック因果のモデル（先読み 0）は CER 11.82 %（CBS RNN-T comp: 16.25 %）、
F/D の F1 85.3 / 57.6 です。発話全体を見る版は 9.52 % です。詳しくはモデルカードをご覧ください。

### 7. 以前のストリーミング Transducer

以前のモデルは素のストリーミング `Speech2Text` を使います。4 の `comp` / `span` モデルも同じように読み込みます。

```python
from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text

s2t = Speech2Text.from_pretrained(
    "fujielab/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132",
    streaming=True, lm_weight=0.0, beam_size=20, beam_search_config=dict(search_type="maes"),
)
# feed 100 ms chunks as in 4; the tokens of each result:
#   s2t.hypotheses_to_results(s2t.streaming_decode(chunk, is_final=is_final))[0][1]
```

スクリプト: `examples/run_streaming_asr.py`（ファイル）、`examples/run_streaming_asr_live.py`（マイク、Silero VAD 付き）、
`examples/demo.py`（Gradio、発話単位）。

## ライセンス

**コード.** このパッケージのコードは Apache License 2.0 です（[LICENSE](LICENSE)）。
学習済みモデルには、コードのライセンスは適用されません。

**学習済みモデルと学習データ.** モデルは次のコーパスで学習しており、その条件がモデルにも及びます。

- **CSJ** （日本語話し言葉コーパス、国立国語研究所）— タグに `csj` を含むモデルと `…_cc0105`。
  国立国語研究所の非商業利用の利用許諾契約によります。商業利用は禁止されています。また現行の契約では、
  コーパスで学習した AI モデルは「派生物」とされ、契約者自身の内部での利用に限られ、第三者には提供できません。
- **CEJC** （日本語日常会話コーパス、国立国語研究所）— タグに `cejc` を含むモデルと `…_cc0105`。
  こちらも国立国語研究所の非商業利用の利用許諾契約によります。商業利用は禁止され、派生物（データ・データベース・
  プログラム）は内部での利用に限られ、第三者には提供できません。コーパスには話者の個人情報が含まれ、
  その公開も禁止されています。
- **ReazonSpeech** （株式会社レアゾン・ホールディングス）— タグに `reazon` を含むモデル。疑似ラベルを付けた
  学習データとして使っています。CDLA-Sharing-1.0 で配布され、利用は著作権法第 30 条の 4 の目的に限られます。

どのモデルも学習データに CSJ と CEJC の少なくとも一方を含むので、すべてのモデルに CSJ・CEJC の条件が及びます。
モデルは商業目的には使えず、研究目的であっても、研究室の外へ提供することはこれらの条件のもとでは難しい場合が
あります。モデルを使った研究成果を発表するときは、このリポジトリと、使ったコーパスを引用してください。

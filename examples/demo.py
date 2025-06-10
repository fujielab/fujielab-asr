# Gradioデモ: マイク録音またはファイルアップロードでASR認識
def main():
    import gradio as gr
    import torch
    import torchaudio
    # from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text
    from espnet2.bin.asr_inference import Speech2Text
    import tempfile
    import os

    model_names = {
        "Japanese/Kana/CombinedToken": "fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105",
        "Japanese/Kanji/CombinedToken": "fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132",
    }

    model_name = "fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132"

    # 推論器の初期化関数
    def create_speech2text(model_name):
        return Speech2Text.from_pretrained(
            model_name,
            streaming=False,
            lm_weight=0.0,
            # beam_size=20,
            # beam_search_config=dict(search_type="maes")
        )

    # グローバル変数でspeech2textを管理
    global speech2text
    speech2text = create_speech2text(model_names[list(model_names.keys())[0]])

    def update_model(model_key):
        global speech2text
        # UIを一時的に無効化するためTrueを返す
        speech2text = create_speech2text(model_names[model_key])
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    def recognize(audio):
        if isinstance(audio, tuple):
            sr, wav = audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, torch.tensor(wav).unsqueeze(0), sr)
                wav_path = tmp.name
        elif isinstance(audio, str):
            wav_path = audio
        else:
            return "Invalid audio input"
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        results = speech2text(wav.squeeze(0))
        if isinstance(results, list) and len(results) > 0:
            text = results[0][0]
        else:
            text = "Unable to recognize speech"
        if 'tmp' in locals():
            os.remove(wav_path)
        return text

    with gr.Blocks() as demo:
        gr.Markdown("""
# Fujie Lab Speech Recognition Demo
This demo lets you try Japanese speech recognition using two different models. You can record your voice or upload an audio file, select a model, and get the transcription result. Please wait for the model to load when switching.
""")
        model_dropdown = gr.Dropdown(
            choices=list(model_names.keys()),
            value=list(model_names.keys())[0],
            label="Model Selection"
        )
        audio_input = gr.Audio(type="numpy", label="Record or Upload Audio")
        output_text = gr.Textbox(label="Recognition Result")
        submit_btn = gr.Button("Run Speech Recognition")

        # モデル切替時はUIを一時的にdisable
        def on_model_change(model_key):
            # UIをdisable
            return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
        model_dropdown.change(on_model_change, inputs=model_dropdown, outputs=[model_dropdown, audio_input, submit_btn], queue=False)
        # モデル切替完了後にenable & speech2text再生成
        model_dropdown.change(update_model, inputs=model_dropdown, outputs=[model_dropdown, audio_input, submit_btn], queue=True)

        submit_btn.click(recognize, inputs=audio_input, outputs=output_text)
        # gr.Markdown("マイクまたはファイルで音声認識を体験できます。モデル切替中はUIが一時的に無効になります。")

    demo.launch(share=True)

if __name__ == "__main__":
    main()

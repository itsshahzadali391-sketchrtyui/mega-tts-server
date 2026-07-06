import gradio as gr
import edge_tts
import asyncio
import httpx
import os
import time
import tempfile

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

# ===== Edge TTS Voices =====
EDGE_VOICES = {
    "Guy (Male)": "en-US-GuyNeural",
    "Aria (Female)": "en-US-AriaNeural",
    "Jenny (Female)": "en-US-JennyNeural",
    "Davis (Male)": "en-US-DavisNeural",
    "Sara (Female)": "en-US-SaraNeural",
    "Tony (Male)": "en-US-TonyNeural",
    "Brian (Male)": "en-US-BrianNeural",
    "Andrew (Male)": "en-US-AndrewNeural",
    "Michelle (Female)": "en-US-MichelleNeural",
    "Adam (Male)": "en-US-AdamNeural",
    "Sam (Male)": "en-US-SamNeural",
}

# ===== TTS Engines =====
TTS_ENGINES = {
    "Edge TTS (Free Unlimited)": "edge",
    "SpeechT5 (Microsoft)": "speecht5",
    "Bark (Suno Expressive)": "bark",
    "VITS (Multilingual)": "vits",
    "MMS (1000+ Languages)": "mms",
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Edge TTS =====
async def generate_edge(text, voice_id, rate, pitch):
    rate_str = f"+{rate}%" if rate >= 0 else f"{rate}%"
    pitch_str = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"
    timestamp = int(time.time() * 1000)
    output_file = os.path.join(OUTPUT_DIR, f"edge_{timestamp}.mp3")
    communicate = edge_tts.Communicate(text, voice_id, rate=rate_str, pitch=pitch_str)
    await communicate.save(output_file)
    return output_file

# ===== HuggingFace TTS =====
def generate_hf_tts(text, model_id, output_format="wav"):
    if not HF_API_TOKEN:
        return None, "HuggingFace token not set!"
    
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            timestamp = int(time.time() * 1000)
            ext = "wav" if output_format == "wav" else "mp3"
            output_file = os.path.join(OUTPUT_DIR, f"hf_{timestamp}.{ext}")
            with open(output_file, "wb") as f:
                f.write(response.content)
            return output_file, "Success!"
        else:
            return None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===== Main TTS Handler =====
def generate_tts(text, engine, voice, rate, pitch):
    if not text.strip():
        return None, "Enter text!"
    
    engine_type = TTS_ENGINES.get(engine, "edge")
    
    try:
        if engine_type == "edge":
            voice_id = EDGE_VOICES.get(voice, "en-US-GuyNeural")
            output = asyncio.run(generate_edge(text, voice_id, rate, pitch))
            return output, f"Edge TTS: Done! Voice: {voice}"
        
        elif engine_type == "speecht5":
            output, msg = generate_hf_tts(text, "microsoft/speecht5_tts")
            return output, f"SpeechT5: {msg}"
        
        elif engine_type == "bark":
            output, msg = generate_hf_tts(text, "suno/bark-small")
            return output, f"Bark: {msg}"
        
        elif engine_type == "vits":
            output, msg = generate_hf_tts(text, "facebook/mms-tts-eng")
            return output, f"VITS: {msg}"
        
        elif engine_type == "mms":
            output, msg = generate_hf_tts(text, "facebook/mms-tts-eng")
            return output, f"MMS: {msg}"
        
        else:
            return None, "Unknown engine!"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# ===== Gradio UI =====
css = ".title{text-align:center;font-size:2.2em;font-weight:bold;color:#2563eb}"

with gr.Blocks(css=css, title="Mega TTS Server", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <div class="title">Mega TTS Server</div>
    <p style="text-align:center;color:#666">5 TTS Engines | Free & Unlimited | No API Key Needed</p>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            tts_text = gr.Textbox(
                label="Text to Speak",
                lines=6,
                placeholder="Type your text here..."
            )
            
            engine_dd = gr.Dropdown(
                choices=list(TTS_ENGINES.keys()),
                value="Edge TTS (Free Unlimited)",
                label="TTS Engine"
            )
            
            voice_dd = gr.Dropdown(
                choices=list(EDGE_VOICES.keys()),
                value="Guy (Male)",
                label="Voice (Edge TTS)"
            )
            
            with gr.Row():
                rate_slider = gr.Slider(-50, 50, value=0, label="Speed")
                pitch_slider = gr.Slider(-50, 50, value=0, label="Pitch")
            
            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Generated Audio", type="filepath")
            status_output = gr.Textbox(label="Status")
    
    gr.Markdown("""
    ---
    ### 5 TTS Engines:
    
    | Engine | Quality | Free | Speed |
    |--------|---------|------|-------|
    | **Edge TTS** | Microsoft Neural | Unlimited | Fast |
    | **SpeechT5** | Microsoft AI | Unlimited | Medium |
    | **Bark** | Suno Expressive | Unlimited | Slow |
    | **VITS** | Facebook Multilingual | Unlimited | Fast |
    | **MMS** | Facebook 1000+ Lang | Unlimited | Fast |
    
    ### How to Use:
    1. Type text
    2. Select engine
    3. Select voice (for Edge TTS)
    4. Click "Generate Speech"
    
    **No API key needed for Edge TTS!**
    """)
    
    generate_btn.click(
        fn=generate_tts,
        inputs=[tts_text, engine_dd, voice_dd, rate_slider, pitch_slider],
        outputs=[audio_output, status_output]
    )

app = demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

#run before running the server
import edge_tts
from openai import OpenAI
import hashlib
import asyncio
import numpy as np
import soundfile as sf
import io
import wave
from scipy import signal
import subprocess
import os
import uvicorn
import sys

sys.stdout.reconfigure(line_buffering=True)


def build_configs():
    edge_compatible_voices = {
    "EN_F": "en-US-JennyNeural",
    "EN_M": "en-US-GuyNeural",
    "AR_F": "ar-SA-ZariyahNeural",
    "AR_M": "ar-SA-HamedNeural",
    "FA_F": "fa-IR-DilaraNeural",
    "FA_M": "fa-IR-FaridNeural"
    }

    configs = {}

    configs["EN_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to English and ONLY respond with the English translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "English",
          "voice_name": edge_compatible_voices["EN_F"]
    }

    configs["EN_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to English and ONLY respond with the English translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "English",
          "voice_name": edge_compatible_voices["EN_M"]
    }

    configs["FA_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Persian and ONLY respond with the Persian translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "Persian",
          "voice_name": edge_compatible_voices["FA_F"]
    }

    configs["FA_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Persian and ONLY respond with the Persian translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "Persian",
          "voice_name": edge_compatible_voices["FA_M"]
    }

    configs["AR_F"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Arabic and ONLY respond with the Arabic translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "Arabic",
          "voice_name": edge_compatible_voices["AR_F"]
    }

    configs["AR_M"] = {
          "system_instruction":"You are a translator. When I send you text, translate it to Arabic and ONLY respond with the Arabic translation. Do not have a conversation, do not ask questions, do not explain, do not hold a conversation.",
          "target_language": "Arabic",
          "voice_name": edge_compatible_voices["AR_M"]
    }

    return configs

def build_clients():
    # Clients
    clients = {}
    history = {}
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    clients["EN"] = client
    clients["AR"] = client
    clients["FA"] = client

    # Based on TARGET language
    history["EN"] = []
    history["AR"] = []
    history["FA"] = []

    return clients, history

def make_cache_key(text, lang, recent_history):
    # Use ONLY previous history, don't include current user input yet
    context_str = "||".join([m["content"] for m in recent_history])
    hash_val = hashlib.md5(context_str.encode()).hexdigest()
    return f"{lang}||{text.strip()}||{hash_val}"

def get_cached_translation(cache_key):
    return translation_cache.get(cache_key)

def add_translation_to_cache(cache_key, value):
    translation_cache[cache_key] = value

async def generate_text_for_lang(k2, sys, text, tgt):
    if k2 not in clients.keys():
        raise ValueError(f"No API key configured for language: {k2}")

    # Switch the API key for this call
    client = clients[k2]
    history_context = histories[k2]

    # Take last N history messages (optional)
    recent_history = history_context[-5:] if history_context else []
    print("***LANG: ", k2, flush=True)
    print("***recent_history: ", recent_history, flush=True)

    # Build cache key
    cache_key = make_cache_key(text, tgt, recent_history)
    print("### Cache Key:", cache_key, flush=True)

    # Check cache
    cached = get_cached_translation(cache_key)
    if cached:
        print("**### cache used!", flush=True)
        return cached

    # Build prompt
    prompt = {"messages": [{"role": "system", "content": sys}] + recent_history + [{"role": "user", "content": text}]}
    print("### prompt:", prompt, flush=True)

    response = client.chat.completions.create(
        model="deepseek-chat",
        input=prompt,
        stream=False
    )

    print("**### API used!", flush=True)
    translated_text = response.choices[0].message.content

    # Update History (maintain max 6 items)
    history_context.append({"role": "user", "content": text})
    history_context.append({"role": "assistant", "content": translated_text})
    if len(history_context) > 5:
        history_context[:] = history_context[-5:]
    print("***NEW history: ", history_context, flush=True)

    # Store in Cache
    add_translation_to_cache(cache_key, translated_text)

    return translated_text

async def gpt_translate(k2, config, text_input):
    transcript_text = ''
    target_sample_rate = 16000

    try:

        if isinstance(config, dict):
            system_instruction = config.get("system_instruction", "Translate this text.")
            target_language = config.get("target_language", "English")
        else:
            print("error: wrong config", flush=True)

        transcript_text = await generate_text_for_lang(k2, system_instruction, text_input, target_language)

        return transcript_text

    except Exception as e:
        print(f"Error in gpt_translate: {e}", flush=True)
        raise

async def tts (text, config):
    try:
        if isinstance(config, dict):
            target_language = config.get("target_language", "English")
            voice_name = config.get("voice_name", "en-US-JennyNeural")
        else:
            print("error: wrong config", flush=True)
        target_sample_rate = 16000
        communicate = edge_tts.Communicate(text=text, voice=voice_name)
        mp3_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
        mp3_data = b"".join(mp3_chunks)
        ffmpeg = subprocess.Popen(
            ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        wav_data, _ = ffmpeg.communicate(input=mp3_data)
        wav_file = io.BytesIO(wav_data)
        with wave.open(wav_file, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            raw_audio = wav.readframes(wav.getnframes())
        if sample_width == 2:
            audio_samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_samples = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        if n_channels == 2:
            audio_samples = audio_samples.reshape(-1, 2).mean(axis=1)

        if sample_rate != target_sample_rate:
            num_samples = int(len(audio_samples) * target_sample_rate / sample_rate)
            audio_samples = signal.resample(audio_samples, num_samples)
            sample_rate = target_sample_rate

        return audio_samples, sample_rate

    except Exception as e:
        print(f"[edge_tts_to_float_audio] Error: {e}", flush=True)
        raise

def get_client_key(tgt_lang, speaker_id):
    return {
        "en-US": "EN_F" if speaker_id == 0 else "EN_M",
        "ar-SA": "AR_F" if speaker_id == 0 else "AR_M",
        "fa-IR": "FA_F" if speaker_id == 0 else "FA_M",
    }.get(tgt_lang) or (_ for _ in ()).throw(ValueError(f"Unsupported target language: {tgt_lang}"))

async def t2S_translate(text_input, tgt_lang, speaker_id):
  # Clients and configs (should differ for each language and speaker)
  key = get_client_key(tgt_lang, speaker_id)
  k2 = key[:2]

  translated_text = await gpt_translate(
    k2=k2,
    config=configs[key],
    text_input=text_input

)
  audio_bytes, sample_rate = await tts(
    text=translated_text,
    config=configs[key]
)
  return audio_bytes, sample_rate, translated_text

# Set up Fast api
import asyncio, time, base64, io
from collections import defaultdict
from fastapi import FastAPI, WebSocket
from typing import Dict, List
from contextlib import asynccontextmanager
from starlette.websockets import WebSocketState
#from file import t2S_translate
current_recorder: WebSocket | None = None

PING_TIMEOUT = 40  # seconds

async def lifespan(app: FastAPI):
    global configs, clients, histories, translation_cache
    translation_cache = {}
    configs = build_configs()
    clients, histories = build_clients()

    print("Starting up ...", flush=True)
    yield
    print("Shutting down. Closing all WebSocket connections...", flush=True)
    websockets = list(rooms[DEFAULT_ROOM].keys())

    for ws in websockets:
        try:
            await ws.close(code=1001)  # 1001 = Going Away
        except Exception as e:
            print(f"WebSocket Disconnected! {e} ", flush=True)
        finally:
            rooms[DEFAULT_ROOM].pop(ws, None)

    print("Shutdown complete.", flush=True)

app = FastAPI(lifespan=lifespan)

DEFAULT_ROOM = "default_room"
rooms: Dict[str, Dict[WebSocket, Dict]] = {
    DEFAULT_ROOM: {}
}

async def translate(src_lang, tgt_lang, text, speaker_id):
    speaker_id = int(speaker_id)
    try:
        audio, sample_rate, translated_text = await t2S_translate(text, tgt_lang, speaker_id)
        if len(audio) == 0:
            print("WARNING: Empty audio received!", flush=True)
            return translated_text, ""

        print(f"Final audio for WAV - shape: {audio.shape}, min/max: {audio.min()}/{audio.max()}", flush=True)

        # Convert float32 normalized audio (-1.0 to 1.0) back to int16 PCM
        int16_audio = (audio * 32767).astype(np.int16)

        # Create WAV in memory buffer
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            n_channels = 1
            sampwidth = 2  # bytes for int16
            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(int16_audio.tobytes())

        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')

        print(f"Base64 audio length: {len(audio_b64)} chars", flush=True)
        return translated_text, audio_b64

    except Exception as e:
        print(f"Error in translate function: {e}", flush=True)
        return f"Translation error: {str(e)}", ""


async def group_translate(connections, src_lang: str, tgt_lang: str, text: str, speaker_id: int):
    translated_text, audio_b64 = await translate(src_lang, tgt_lang, text, speaker_id)

    for ws in connections:
        try:
            await ws.send_json({
              "type" : "translate_msg",
              "transcript": text,
              "translated_text": translated_text,
              "translated_audio_url": audio_b64,
              "src_lang": src_lang,
              "tgt_lang": tgt_lang,
            })
            print(f"WebSocket x recieved src_lang: {src_lang}, tgt_lang: {tgt_lang}", flush=True)
        except Exception as e:
          print(f"Error translating for group {tgt_lang}/{speaker_id}: {e}", flush=True)

async def just_send(ws: WebSocket, src_lang: str, text: str):

    try:
        await ws.send_json({
            "type" : "transcript_msg",
            "transcript": text,
            "src_lang": src_lang
        })
        print(f"WebSocket x recieved src_lang: {src_lang}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

async def per_record(connections, per: bool):
    for ws in connections:
        try:
            await ws.send_json({
                "type" : "per_record",
                "per_record": per,
            })
            print(f"per_record: {per}", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global current_recorder

    # Send current per_record state to the new user
    await websocket.send_json({
        "type": "per_record",
        "per_record": current_recorder is None  # True => اجازه رکورد هست
    })

    # Set default values
    user_data = {
        "lang": "en-US",
        "speaker_id": "0",
        "last_ping": time.time()
    }

    # Add user to default room
    rooms[DEFAULT_ROOM][websocket] = user_data
    last_active = time.time()

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                data = await websocket.receive_json()
                last_active = time.time()
                # global current_recorder

                if data.get("type") == "update_settings":
                    lang = data.get("lang", "en-US")
                    speaker_id = int(data.get("speaker_id", "0"))
                    rooms[DEFAULT_ROOM][websocket]["lang"] = lang
                    rooms[DEFAULT_ROOM][websocket]["speaker_id"] = speaker_id
                    await websocket.send_json({"status": "settings_updated"})
                    print(f"WebSocket {websocket} updated settings: lang={lang}, speaker_id={speaker_id}", flush=True)

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    user_data["last_ping"] = last_active

                elif data.get("type") == "speak":
                    src_lang = data.get("src_lang")
                    text = data.get("text")
                    speaker_id = int(data.get("speaker_id", "0"))
                    print(f"ws x , src_lang: {src_lang} *type speak", flush=True)
                    if not src_lang or not text:
                        continue

                    # Group users by their language + speaker_id
                    groups = defaultdict(list)
                    for ws, info in rooms[DEFAULT_ROOM].items():
                        key = (info["lang"], info["speaker_id"])
                        groups[key].append(ws)

                    tasks = []
                    for (tgt_lang, speaker_id), connections in groups.items():
                        if src_lang == tgt_lang:
                            for ws in connections:
                                tasks.append(
                                    just_send(ws, src_lang, text)
                                    )
                        else:
                            tasks.append(
                                group_translate(connections, src_lang, tgt_lang, text, speaker_id)
                            )
                    await asyncio.gather(*tasks)
                elif data.get("type") == "status_Record":
                    if data.get("statusRecord") == True:
                        await per_record(list(rooms[DEFAULT_ROOM].keys()), False)
                        current_recorder = websocket
                    elif data.get("statusRecord") == False:
                        await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
                        current_recorder = None


            except Exception as e:
                print(f"Client error: {e}", flush=True)
                break

            if time.time() - last_active > PING_TIMEOUT:
                print("Client inactive, disconnecting.", flush=True)
                break

    except Exception as e:
        print(f"Connection error: {e}", flush=True)

    finally:
        rooms[DEFAULT_ROOM].pop(websocket, None)

        # همیشه بررسی کنیم که اگر این کاربر رکوردر بود، آن را خالی کنیم
        if current_recorder == websocket:
            await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
            current_recorder = None

        # تلاش برای بستن سوکت
        try:
            await websocket.close()
        except Exception as e:
            print(f"Error closing WebSocket: {e}", flush=True)

    # finally:
    #     rooms[DEFAULT_ROOM].pop(websocket, None)
    #     if websocket.client_state != WebSocketState.DISCONNECTED:
    #         if current_recorder == websocket:
    #             await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
    #             current_recorder = None
    #         try:
    #             await websocket.close()
    #         except Exception as e:
    #             print(f"Error closing WebSocket: {e}")

# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8258)

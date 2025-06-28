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
import re
import logging

sys.stdout.reconfigure(line_buffering=True)

log_file = "./app.log"

if not os.path.exists(log_file):
    open(log_file, "a").close()  

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)  

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

file_only_logger = logging.getLogger("file_only")
file_only_logger.setLevel(logging.DEBUG)
file_only_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into English."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
            ),
        "target_language": "English",
        "voice_name": edge_compatible_voices["EN_F"]
    }

    configs["EN_M"] = {
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into English."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
                "example: [{role}:{user}, {content}:{[tr]سلام وقت بخیر[\tr]},{role}:{assistant}, {content}:مرحبا وقتکم سعید]"
            ),
        "target_language": "English",
        "voice_name": edge_compatible_voices["EN_M"]
    }

    configs["FA_F"] = {
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into Persian."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
                "example: [{role}:{user}, {content}:{[tr]سلام وقت بخیر[\tr]},{role}:{assistant}, {content}:مرحبا وقتکم سعید]"
            ),
        "target_language": "Persian",
        "voice_name": edge_compatible_voices["FA_F"]
    }

    configs["FA_M"] = {
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into Persian."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
            ),
        "target_language": "Persian",
        "voice_name": edge_compatible_voices["FA_M"]
    }

    configs["AR_F"] = {
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into Arabic."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
            ),
        "target_language": "Arabic",
        "voice_name": edge_compatible_voices["AR_F"]
    }

    configs["AR_M"] = {
        "system_instruction":(
                "You are a natural translation bot. ONLY TRANSLATE the text found between [tr] and [/tr] tags from any language into Arabic."
                "Do not reply with anything else. Only output the natural translation without any tags or extra text."
                "Do not explain or add context or notes."
                "If the input contains no [tr] tags, return nothing."
                "If the input has any problems, output the most likely translation."
            ),
        "target_language": "Arabic",
        "voice_name": edge_compatible_voices["AR_M"]
    }

    return configs

def build_clients():
    # Clients
    clients = {}
    history = {}
    client = OpenAI(api_key=os.environ["Open_AI"])
    client1 = OpenAI(api_key=os.environ["Open_AI"])
    client2 = OpenAI(api_key=os.environ["Open_AI"])

    clients["EN"] = client
    clients["AR"] = client1
    clients["FA"] = client2

    # Based on TARGET language
    history["EN"] = []
    history["AR"] = []
    history["FA"] = []

    return clients, history

def clean_model_output(output: str) -> str:
    return re.sub(r"\[/?tr\]", "", output.strip())

async def generate_text_for_lang(k2, sys, text, tgt):
    global clients, histories
    if k2 not in clients.keys():
        raise ValueError(f"No API key configured for language: {k2}")

    client = clients[k2]
    history_context = histories[k2]

    prompt = [{"role": "developer", "content": sys}]
    prompt += history_context
    prompt.append({"role": "user", "content": "[tr]" + text + "[\tr]"})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt
    )
    file_only_logger.info("***Prompt: %s", prompt)

    translated_text = clean_model_output(response.choices[0].message.content)

    file_only_logger.info("***Translate: %s", translated_text)

    history_context.append({"role": "user", "content": "[tr]" + text + "[\tr]"})
    history_context.append({"role": "assistant", "content": translated_text})

    max_history_messages = 6  # 3 pairs
    if len(history_context) > max_history_messages:
        history_context[:] = history_context[-max_history_messages:]

    return translated_text

async def gpt_translate(k2, config, text_input):
    transcript_text = ''
    target_sample_rate = 16000

    try:
        if isinstance(config, dict):
            
            target_language = config.get("target_language", "English")
            system_instruction = config.get("system_instruction", "Translate the text in [tr] tag to {target_language}")
        else:
            logger.info("Error: Wrong config!")
            
        logger.info("***Target language: %s", target_language)

        transcript_text = await generate_text_for_lang(k2, system_instruction, text_input, target_language)

        return transcript_text

    except Exception as e:
        logger.exception(f"Error in gpt_translate: {e}")
        raise

async def tts (text, config):
    try:
        if isinstance(config, dict):
            target_language = config.get("target_language", "English")
            voice_name = config.get("voice_name", "en-US-JennyNeural")
        else:
            logger.info("Error: Wrong config!")
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
        logger.exception(f"[edge_tts_to_float_audio] Error: {e}")
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

PING_TIMEOUT = 10  # seconds

async def lifespan(app: FastAPI):
    global configs, clients, histories
    configs = build_configs()
    clients, histories = build_clients()

    logger.info("Starting up ...")
    yield
    logger.info("Shutting down. Closing all WebSocket connections...")
    websockets = list(rooms[DEFAULT_ROOM].keys())

    for ws in websockets:
        try:
            await ws.close(code=1001)  # 1001 = Going Away
        except Exception as e:
            logger.exception(f"WebSocket Disconnected! {e} ")
        finally:
            rooms[DEFAULT_ROOM].pop(ws, None)

    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

DEFAULT_ROOM = "default_room"
rooms: Dict[str, Dict[WebSocket, Dict]] = {
    DEFAULT_ROOM: {}
}

async def translate(src_lang, tgt_lang, text, speaker_id):
    speaker_id = int(speaker_id)
    try:
        logger.info("Source language: %s", src_lang)
        audio, sample_rate, translated_text = await t2S_translate(text, tgt_lang, speaker_id)

        if len(audio) == 0:
            logger.info("WARNING: Empty audio received!")
            return translated_text, ""

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

        return translated_text, audio_b64

    except Exception as e:
        logger.exception(f"Error in translate function: {e}")
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
            logger.info(f"WebSocket recieved src_lang: {src_lang}, tgt_lang: {tgt_lang}")
        except Exception as e:
          logger.exception(f"Error translating for group {tgt_lang}/{speaker_id}: {e}")

async def just_send(ws: WebSocket, src_lang: str, text: str):

    try:
        await ws.send_json({
            "type" : "transcript_msg",
            "transcript": text,
            "src_lang": src_lang
        })
    except Exception as e:
        logger.exception(f"Error: {e}")

async def per_record(connections, per: bool):
    a = 0
    for ws in connections:
        try:
            await ws.send_json({
                "type" : "per_record",
                "per_record": per,
            })
            if a == 0: 
                logger.info(f"per_record: {per}")
                a = 1
        except Exception as e:
            logger.exception(f"Error: {e}")

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
                    logger.info(f"WebSocket {websocket} updated settings: lang={lang}, speaker_id={speaker_id}")

                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    user_data["last_ping"] = last_active

                elif data.get("type") == "speak":
                    src_lang = data.get("src_lang")
                    text = data.get("text")
                    speaker_id = int(data.get("speaker_id", "0"))
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
                logger.exception(f"Client error: {e}")
                break

            if time.time() - last_active > PING_TIMEOUT:
                logger.info(f"Client inactive for {PING_TIMEOUT} seconds, disconnecting.")

                if current_recorder == websocket:
                    await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
                    current_recorder = None
                    logger.info("Recorder auto-released due to timeout.")

                break

    except Exception as e:
        logger.exception(f"Connection error: {e}")

    finally:
        rooms[DEFAULT_ROOM].pop(websocket, None)

        if current_recorder == websocket:
            await per_record(list(rooms[DEFAULT_ROOM].keys()), True)
            current_recorder = None

        try:
            await websocket.close()
        except Exception as e:
            logger.exception(f"Error closing WebSocket: {e}")


# Run the FastAPI app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8258)

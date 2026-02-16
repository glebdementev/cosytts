import json
import os
import time
import urllib.error
import urllib.request
import wave


def fail(message: str) -> None:
    raise RuntimeError(message)


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(repo_root, "deploy", "config.json")
if not os.path.exists(config_path):
    fail(f"Missing deploy config: {config_path}.")

with open(config_path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

host = cfg["ssh"]["host"]
port = int(os.getenv("TTS_API_PORT", "8093"))

base_url = f"http://{host}:{port}"
health_url = f"{base_url}/health"
tts_stream_url = f"{base_url}/synthesize/stream"

output_path = os.path.join(repo_root, "deploy", "tts_test_output.wav")

print(f"Testing health endpoint: {health_url}")
with urllib.request.urlopen(health_url) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("status") != "ok":
    fail(f"Health check failed: {payload}")
print("Health OK")

test_text = "Я ведаю и веду тебя в замок чтоб повесить на замок."
voice_name = os.getenv("TTS_TEST_VOICE", "polina")

request_payload = {"text": test_text, "voice": voice_name}
print(f"Using voice profile: {voice_name}")

# Streaming endpoint (real TTFB)
print("\n" + "=" * 50)
print("Streaming /synthesize/stream (save as WAV)")
print("=" * 50)
request_body = json.dumps(request_payload).encode("utf-8")

print(f"Requesting streaming synthesis: {tts_stream_url}")
request = urllib.request.Request(
    tts_stream_url,
    data=request_body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    start_time = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        sample_rate = int(response.headers.get("X-Sample-Rate", "24000"))
        channels = int(response.headers.get("X-Channels", "1"))
        sample_width = int(response.headers.get("X-Sample-Width", "2"))
        first_chunk = response.read(sample_width)
        if first_chunk:
            ttfb_ms = (time.perf_counter() - start_time) * 1000
            print(f"TTFB (streaming): {ttfb_ms:.2f} ms")
        audio_frames = first_chunk + response.read()
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    fail(f"TTS stream request failed: {exc.code} {exc.reason}. Body: {body}")

if not audio_frames:
    fail("Streaming response contained no audio bytes.")

with wave.open(output_path, "wb") as handle:
    handle.setnchannels(channels)
    handle.setsampwidth(sample_width)
    handle.setframerate(sample_rate)
    handle.writeframes(audio_frames)

size = os.path.getsize(output_path)
print(f"Stream OK - wrote {size} bytes to {output_path}")

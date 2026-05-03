import json
import requests
from pathlib import Path

BASE = "https://acwz-tealeafdetection.hf.space"


# -----------------------------
# Helper: extract image URL safely
# -----------------------------
def extract_image_url(data):
    if isinstance(data, str):
        if "gradio_api/file" in data:
            return data
        return None

    if isinstance(data, list):
        for item in data:
            result = extract_image_url(item)
            if result:
                return result

    if isinstance(data, dict):
        for v in data.values():
            result = extract_image_url(v)
            if result:
                return result

    return None


# -----------------------------
# Upload + Predict + Stream
# -----------------------------
def run_inference(source, confidence=0.25):

    # 1. Upload image
    if hasattr(source, "read"):
        filename = getattr(source, "name", "image.png")
        upload_response = requests.post(
            f"{BASE}/gradio_api/upload",
            files={"files": (filename, source)},
            timeout=120,
        )
    else:
        with open(source, "rb") as f:
            upload_response = requests.post(
                f"{BASE}/gradio_api/upload",
                files={"files": (Path(source).name, f)},
                timeout=120,
            )

    upload_response.raise_for_status()
    file_path = upload_response.json()[0]

    # 2. Start prediction
    prediction_response = requests.post(
        f"{BASE}/gradio_api/call/v2/predict",
        json={
            "image": {
                "path": file_path,
                "meta": {"_type": "gradio.FileData"},
            },
            "confidence": confidence,
        },
        timeout=120,
    )

    prediction_response.raise_for_status()
    event_id = prediction_response.json()["event_id"]

    # 3. Stream results
    stream_url = f"{BASE}/gradio_api/call/predict/{event_id}"
    stream = requests.get(stream_url, stream=True, timeout=120)
    stream.raise_for_status()

    final_data = None
    event_type = None

    for line in stream.iter_lines(decode_unicode=True):
        if not line:
            continue

        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()

        elif line.startswith("data:"):
            raw = line.split(":", 1)[1].strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw

            if event_type == "complete":
                final_data = parsed
                break

    if final_data is None:
        raise RuntimeError("No complete event received from Space")

    # 4. Extract results
    annotated_image_url = extract_image_url(final_data)

    detections = None
    if isinstance(final_data, list):
        for item in final_data:
            if isinstance(item, dict):
                detections = item
                break

    return {
        "annotated_image_url": annotated_image_url,
        "detections": detections,
        "raw": final_data,
    }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    result = run_inference("test.png", confidence=0.25)

    print("\n=== ANNOTATED IMAGE URL ===")
    print(result["annotated_image_url"])

    print("\n=== DETECTIONS ===")
    print(result["detections"])
# tmux attach -t api


import os
import sys
import tempfile
from datetime import datetime
import scipy
import numpy as np
scipy.inf = np.inf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from app import initialize_models, process_audio, rule_post_processing
from app import AFTER_DOWNSAMPLING_FRAME_RATES
import uvicorn

app = FastAPI(title="SongFormer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

models_ready = False
process_audio_func = None
postprocess_func = None


def init_models():
    # Download SongFormer models
    global models_ready, process_audio_func, postprocess_func
    
    initialize_models("SongFormer", "SongFormer.safetensors", "SongFormer.yaml")

    
    process_audio_func = process_audio
    postprocess_func = rule_post_processing
    models_ready = True



def format_time(sec):
    return f"{int(sec//60):02d}:{sec%60:06.3f}"


def to_segments(msa):
    # Converts [(time, label), ...] into a list of segments
    segments = []
    for i in range(len(msa) - 1):
        start, label = msa[i]
        end = msa[i + 1][0]
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "time": f"{format_time(start)} → {format_time(end)}",
            "duration": round(end - start, 2),
            "label": label
        })
    return segments


def logits_to_dict(logits, topk=8):
    # Converts torch logits into a JSON-friendly dict.
    # Returns:
    #   - time_sec: time axis
    #   - boundary_logits: list of length T
    #   - function_logits_topk: top-k classes (id + values)

    function_vals = logits["function_logits"].squeeze(0).detach().cpu().numpy()  # [T, C]
    boundary_vals = logits["boundary_logits"].squeeze(0).detach().cpu().numpy()  # [T]
    T, C = function_vals.shape

    # frame rate
    try:
        
        fr = float(AFTER_DOWNSAMPLING_FRAME_RATES)
    except Exception:
        fr = 8.333

    time_sec = (np.arange(T) / fr).tolist()

    # id->label 
    try:
        from dataset.label2id import ID_TO_LABEL
    except Exception:
        ID_TO_LABEL = {i: f"Class_{i}" for i in range(C)}

    # top-k on mean activation
    top_classes = np.argsort(function_vals.mean(axis=0))[-topk:][::-1]

    function_topk = []
    for cls in top_classes:
        function_topk.append({
            "id": int(cls),
            "name": ID_TO_LABEL.get(int(cls), f"Class_{int(cls)}"),
            "values": function_vals[:, int(cls)].astype(np.float32).tolist()
        })

    return {
        "frame_rate": fr,
        "T": int(T),
        "C": int(C),
        "time_sec": time_sec,
        "boundary_logits": boundary_vals.astype(np.float32).tolist(),
        "function_logits_topk": function_topk
    }



@app.get("/")
def home():
    return {"api": "SongFormer", "docs": "/docs", "ready": models_ready}


@app.get("/health")
def health():
    return {
        "status": "ok" if models_ready else "loading",
        "gpu": torch.cuda.is_available()
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    
    if not models_ready:
        raise HTTPException(503, "model load error")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(400, f"Формат {ext} не підтримується")
    

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        t0 = datetime.now()
        

        logits, msa = process_audio_func(tmp_path)
        # _, msa = process_audio_func(tmp_path)
        msa = postprocess_func(msa)
        
        segments = to_segments(msa)
        labels = [s["label"] for s in segments]
        print(logits)
        return {
            "filename": file.filename,
            "processing_sec": round((datetime.now() - t0).total_seconds(), 2),
            "segments": segments,
            "structure": "-".join(labels),
            "logits": logits_to_dict(logits, topk=8),
            "stats": {
                "count": len(segments),
                "duration": segments[-1]["end"] if segments else 0,
                "labels": {l: labels.count(l) for l in set(labels)}
            }
        }
    finally:
        os.unlink(tmp_path)



if __name__ == "__main__":


    init_models()
    
    print("\nhttp://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
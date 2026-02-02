import os
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

UPLOAD_DIR = Path("api_uploads")
OUTPUT_DIR = Path("api_outputs")
MIDIFREN_SCRIPT = "MIDIfren.py"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="MIDIfren API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SoundType(str, Enum):
    vocals = "vocals"
    melody = "melody"
    drums = "drums"
    bass = "bass"


class ProcessResponse(BaseModel):
    job_id: str
    message: str
    output_files: List[str]
    download_urls: List[str]


def run_midifren(
    input_path: Path,
    job_output_dir: Path,
    sound_type: str,
    extract_stem: bool,
    convert_midi: bool,
    quantize: bool,
    pitchbend: bool,
    bpm: Optional[int],
    onset: Optional[float],
    note_length: Optional[float],
    groove: Optional[str],
) -> tuple[bool, str, List[str]]:
    """Launching MIDIfren CLI"""
    
    cmd = ["python", MIDIFREN_SCRIPT, "-i", str(input_path), "-t", sound_type]
    
    if extract_stem:
        cmd.append("--stem")
    if convert_midi:
        cmd.append("--midi")
    if quantize:
        cmd.append("--quantize")
    if pitchbend:
        cmd.append("--pitchbend")
    if bpm:
        cmd.extend(["-b", str(bpm)])
    if onset is not None:
        cmd.extend(["-o", str(onset)])
    if note_length:
        cmd.extend(["-n", str(note_length)])
    if groove:
        cmd.extend(["-g", groove])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        #  MIDIfren outputs to the “output” folder
        output_files = []
        midifren_output = Path("output")
        
        if midifren_output.exists():
            for f in midifren_output.iterdir():
                if f.is_file():
                    dest = job_output_dir / f.name
                    shutil.move(str(f), str(dest))
                    output_files.append(f.name)
        
        if result.returncode == 0:
            return True, "OK", output_files
        else:
            return False, result.stderr or result.stdout or "Unknown error", output_files
            
    except subprocess.TimeoutExpired:
        return False, "Timeout (10 min)", []
    except Exception as e:
        return False, str(e), []


async def cleanup_later(job_id: str, hours: int = 24):
    import asyncio
    await asyncio.sleep(hours * 3600)
    shutil.rmtree(UPLOAD_DIR / job_id, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR / job_id, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sound_type: SoundType = Form(...),
    extract_stem: bool = Form(False),
    convert_midi: bool = Form(False),
    quantize: bool = Form(False),
    pitchbend: bool = Form(False),
    bpm: Optional[int] = Form(None),
    onset: Optional[float] = Form(None),
    note_length: Optional[float] = Form(None),
    groove: Optional[str] = Form(None),
):

#     Audio file processing.
    
#     - **file**: audio file (wav, mp3, flac), max 100MB
#     - **sound_type**: vocals / melody / drums / bass
#     - **extract_stem**: stem extraction
#     - **convert_midi**: conversion to MIDI
#     - **quantize**: MIDI quantization
#     - **pitchbend**: pitchbend detection
#     - **bpm**: tempo (auto-detection if not specified)
#     - **onset**: sensitivity (0-1)
# - **note_length**: min. note length
# - **groove**: time signature (e.g. ‘4/4’)

    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac"}:
        raise HTTPException(400, "Format not supported. Allowed: wav, mp3, flac")
    
    file.file.seek(0, 2)
    if file.file.tell() > 100 * 1024 * 1024:
        raise HTTPException(400, "The file is too large (max 100MB)")
    file.file.seek(0)
    
    if not extract_stem and not convert_midi:
        raise HTTPException(400, "Specify extract_stem=true and/or convert_midi=true")
    
    # Creating directories
    job_id = str(uuid.uuid4())
    job_upload = UPLOAD_DIR / job_id
    job_output = OUTPUT_DIR / job_id
    job_upload.mkdir(parents=True)
    job_output.mkdir(parents=True)
    
    # Saving the file
    input_path = job_upload / file.filename
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Processing
    success, msg, output_files = run_midifren(
        input_path, job_output, sound_type.value,
        extract_stem, convert_midi, quantize, pitchbend,
        bpm, onset, note_length, groove
    )
    
    if not success:
        shutil.rmtree(job_upload, ignore_errors=True)
        shutil.rmtree(job_output, ignore_errors=True)
        raise HTTPException(500, f"{msg}")
    
    # Self-cleaning after 24 hours
    background_tasks.add_task(cleanup_later, job_id, 24)
    
    return ProcessResponse(
        job_id=job_id,
        message="Готово!",
        output_files=output_files,
        download_urls=[f"/download/{job_id}/{f}" for f in output_files],
    )


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    #Uploading the processed file
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    
    media = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", 
             ".mid": "audio/midi", ".midi": "audio/midi"}
    
    return FileResponse(path, filename=filename, 
                        media_type=media.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/jobs/{job_id}")
async def job_files(job_id: str):
    path = OUTPUT_DIR / job_id
    if not path.exists():
        raise HTTPException(404, "job not found")
    
    files = [{"name": f.name, "size": f.stat().st_size, "url": f"/download/{job_id}/{f.name}"} 
             for f in path.iterdir() if f.is_file()]
    return {"job_id": job_id, "files": files}



if __name__ == "__main__":
    import uvicorn
    print("\nMIDIfren API Server")
    print("http://localhost:9000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=9000)

import os
import sys
from pathlib import Path

# Add src/ to path
src_dir = Path(__file__).resolve().parent
sys.path.append(str(src_dir))

from config_loader import load_config
from audio_io import download_resources
from text_normalization import parse_subtitles, normalize_azerbaijani

def main():
    print("Loading configuration...")
    config = load_config(None)
    
    # Use the local project directory for execution
    project_root = "/home/rufiz/GoogleDrive/VAD"
    raw_audio_path = os.path.join(project_root, "data/raw/audio.wav")
    raw_caption_dir = os.path.join(project_root, "data/raw")
    
    print("--- Stage 2: Downloading audio and subtitles via yt-dlp ---")
    download_resources(
        url=config.video_url,
        output_audio_path=raw_audio_path,
        output_caption_dir=raw_caption_dir
    )
    
    print("\n--- Stage 3: Parsing caption timestamps ---")
    caption_files = [f for f in os.listdir(raw_caption_dir) if f.startswith("captions.") and f.endswith((".vtt", ".srt"))]
    if not caption_files:
        raise FileNotFoundError("No subtitle file downloaded. Check yt-dlp logs.")
        
    caption_path = os.path.join(raw_caption_dir, caption_files[0])
    raw_segments = parse_subtitles(caption_path)
    print(f"Successfully parsed {len(raw_segments)} raw caption segments.")
    
    full_reference_transcript = " ".join([seg[2] for seg in raw_segments])
    
    print("\n--- Stage 4: Azerbaijani Text Normalization (Turkic Case-Folding) ---")
    normalized_transcript = normalize_azerbaijani(full_reference_transcript)
    
    processed_dir = os.path.join(project_root, "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    normalized_path = os.path.join(processed_dir, "transcript_normalized.txt")
    
    with open(normalized_path, "w", encoding="utf-8") as f:
        f.write(normalized_transcript)
        
    print(f"\nLocal data preparation complete!")
    print(f"  - Audio file  : {raw_audio_path} (WAV 16kHz mono)")
    print(f"  - Transcript  : {normalized_path}")
    print(f"Files are now fully prepared and will automatically sync to Google Drive.")

if __name__ == "__main__":
    main()

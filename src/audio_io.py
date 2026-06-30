import os
import yt_dlp
import soundfile as sf

def download_resources(url: str, output_audio_path: str, output_caption_dir: str):
    """Downloads video audio and converts to 16kHz mono WAV, and downloads Azerbaijani subtitles."""
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    os.makedirs(output_caption_dir, exist_ok=True)

    # Base template name without extension for yt-dlp to output to
    audio_base = os.path.splitext(output_audio_path)[0]

    # Check if local static ffmpeg is present in venv to override system path
    local_bin = os.path.expanduser("~/env_vad/bin")
    ffmpeg_loc = local_bin if os.path.exists(os.path.join(local_bin, "ffmpeg")) else None

    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': [
            '-ac', '1',
            '-ar', '16000'
        ],
        'outtmpl': audio_base,
        'quiet': False,
        'ffmpeg_location': ffmpeg_loc,
    }

    print(f"Downloading audio from {url} to {output_audio_path}...")
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])
    
    # Check if yt-dlp appended an extension and rename to match output_audio_path exactly
    if not os.path.exists(output_audio_path):
        if os.path.exists(audio_base + ".wav"):
            os.rename(audio_base + ".wav", output_audio_path)
        elif os.path.exists(audio_base + ".wav.wav"):
            os.rename(audio_base + ".wav.wav", output_audio_path)

    # Download subtitles (captions)
    ydl_opts_subs = {
        'writeprompt': False,
        'skip_download': True,
        'writesubtitles': True,
        'subtitleslangs': ['az'],
        'outtmpl': os.path.join(output_caption_dir, 'captions'),
        'quiet': False,
        'ffmpeg_location': ffmpeg_loc,
    }

    print(f"Downloading subtitles from {url} to {output_caption_dir}...")
    with yt_dlp.YoutubeDL(ydl_opts_subs) as ydl:
        ydl.download([url])


def slice_audio(input_wav: str, output_wav: str, start_time: float, end_time: float):
    """Extracts an audio slice from input_wav and writes it to output_wav using soundfile."""
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    with sf.SoundFile(input_wav) as f:
        sr = f.samplerate
        start_frame = int(start_time * sr)
        end_frame = int(end_time * sr)
        num_frames = max(0, end_frame - start_frame)
        
        f.seek(start_frame)
        data = f.read(num_frames)
        sf.write(output_wav, data, sr)

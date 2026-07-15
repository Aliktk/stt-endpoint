# recorder.py
# Simple voice recorder. Records from your microphone and saves it as WAV or MP3.
#
# Install:
#   pip install sounddevice soundfile numpy
#   # only needed for MP3 output:
#   pip install pydub          (and ffmpeg installed on the system)
#
# Usage:
#   python recorder.py                   record until you press Enter, save WAV
#   python recorder.py --duration 10     record for 10 seconds
#   python recorder.py --format mp3      save as MP3 instead of WAV
#   python recorder.py --name mynote     pick the file name yourself

import argparse
import os
import queue
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 44100   # Hz
CHANNELS = 1          # mono, which is enough for voice


def record_fixed(duration):
    """Record for a fixed number of seconds."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()   # wait until the recording is finished
    return audio


def record_until_enter():
    """Record until the user presses Enter."""
    print("Recording... press Enter to stop.")
    frames = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.put(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="int16", callback=callback)
    with stream:
        input()   # blocks here until Enter is pressed

    # collect everything the mic captured into one array
    chunks = []
    while not frames.empty():
        chunks.append(frames.get())

    if not chunks:
        return np.empty((0, CHANNELS), dtype="int16")
    return np.concatenate(chunks, axis=0)


def save_audio(audio, path, fmt):
    """Save the recording. WAV is written directly, MP3 is converted with ffmpeg."""
    if fmt == "wav":
        sf.write(path, audio, SAMPLE_RATE)
    else:  # mp3
        from pydub import AudioSegment
        tmp_wav = path.replace(".mp3", ".tmp.wav")
        sf.write(tmp_wav, audio, SAMPLE_RATE)
        AudioSegment.from_wav(tmp_wav).export(path, format="mp3")
        os.remove(tmp_wav)


def main():
    ap = argparse.ArgumentParser(description="Record a voice note and save it.")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record; leave out to record until you press Enter")
    ap.add_argument("--format", choices=["wav", "mp3"], default="wav",
                    help="output format (default: wav)")
    ap.add_argument("--name", default=None,
                    help="file name without extension (default: a timestamp)")
    args = ap.parse_args()

    # save recordings in a 'recordings' folder next to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "recordings")
    os.makedirs(out_dir, exist_ok=True)

    name = args.name or datetime.now().strftime("voice_%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{name}.{args.format}")

    # record
    try:
        if args.duration:
            audio = record_fixed(args.duration)
        else:
            audio = record_until_enter()
    except KeyboardInterrupt:
        print("\nStopped.")
        return
    except Exception as e:
        print(f"error while recording: {e}", file=sys.stderr)
        sys.exit(1)

    if len(audio) == 0:
        print("Nothing was recorded.")
        return

    # save
    try:
        save_audio(audio, out_path, args.format)
    except Exception as e:
        print(f"error while saving: {e}", file=sys.stderr)
        sys.exit(1)

    seconds = len(audio) / SAMPLE_RATE
    print(f"Saved {seconds:.1f}s to: {out_path}")


if __name__ == "__main__":
    main()
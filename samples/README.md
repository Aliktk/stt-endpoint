# Samples

Record your own audio to test the pipeline. `audio_creation.py` captures from
your microphone and saves a WAV (or MP3) into `samples/recordings/`.

Install the recording extra once:

```bash
uv sync --extra record
```

Then record:

```bash
# record until you press Enter
uv run python samples/audio_creation.py

# record a fixed length
uv run python samples/audio_creation.py --duration 10

# choose the name / format
uv run python samples/audio_creation.py --name my_note --format mp3
```

Transcribe what you recorded (with the API running):

```bash
curl -F "file=@samples/recordings/my_note.wav" http://localhost:8080/v1/transcribe
```

Recordings are git-ignored — they stay on your machine.

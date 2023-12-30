from faster_whisper import WhisperModel

model_size = "tiny"
model = WhisperModel(model_size, device="cpu",
                     compute_type="int8", download_root="/tmp")
# warm up model
segments, info = model.transcribe("test_audio.mp3", beam_size=5)
print(segments, info)
print("finish")

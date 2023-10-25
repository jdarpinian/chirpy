from faster_whisper import WhisperModel
import librosa
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


filename = 'audio_output_12.wav'
audio, _ = librosa.load(filename, sr=16000)
# TODO: set download_root and/or change model name to path to predownloaded model
model = WhisperModel("base.en", device="cuda", compute_type="float16", download_root=None)

# lots of good options to tweak listed in this debug log: [<generator object WhisperModel.generate_segments at 0x0000016626DB4F10>, TranscriptionInfo(language='en', language_probability=1, duration=39.68, duration_after_vad=39.68, all_language_probs=None, transcription_options=TranscriptionOptions(beam_size=5, best_of=5, patience=1, length_penalty=1, repetition_penalty=1, no_repeat_ngram_size=0, log_prob_threshold=-1.0, no_speech_threshold=0.6, compression_ratio_threshold=2.4, condition_on_previous_text=False, prompt_reset_on_temperature=0.5, temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], initial_prompt=None, prefix=None, suppress_blank=True, suppress_tokens=[-1], without_timestamps=False, max_initial_timestamp=1.0, word_timestamps=True, prepend_punctuations='"\'“¿([{-', append_punctuations='"\'.。,，!！?？:：”)]}、'), vad_options=None)]

for length in range(0, len(audio), 1600*1):
    truncated_audio = audio[0:length][-(16000*10):]
    print( truncated_audio)
    transcription, info = model.transcribe(truncated_audio, language="en", beam_size=5, word_timestamps=False, condition_on_previous_text=False)

    # print(info)
    print(f"{length / 16000.}s: ", end="")

    for segment in transcription:

        print(f"{round(segment.no_speech_prob, 2)}, {round(segment.start, 1)}, {round(segment.end, 1)}: {segment.text}", end=" ") if segment.no_speech_prob < 0.2 else None
    print()

import multiprocessing
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL) # exit instantly on ctrl-c instead of throwing KeyboardInterrupt because it's not reliable, many blocking calls fail to be interrupted by KeyboardInterrupt

def die_if_parent_process_dies_on_linux():
    try:
        import pyprctl
        pyprctl.set_pdeathsig(signal.SIGKILL)
    except ImportError:
        pass

def play_voice_clips_process(voice_clips_queue, stop_speaking_event, audio_start_time):
    die_if_parent_process_dies_on_linux()
    import sounddevice as sd
    import time
    import numpy as np
    while True:
        (voice_clip, last_word_time) = voice_clips_queue.get()
        if voice_clip is None:
            break
        silence_threshold = 0.01
        original_length = len(voice_clip)
        voice_clip = np.trim_zeros(voice_clip, 'f')
        while len(voice_clip) > 0 and abs(voice_clip[0]) < silence_threshold:
            voice_clip = voice_clip[1:]
        # Remove the start of voice_clip up to the next zero crossing
        zero_crossings = np.where(np.diff(np.sign(voice_clip)))[0]
        if len(zero_crossings) > 0:
            voice_clip = voice_clip[zero_crossings[0]:]
        final_len = len(voice_clip)
        # print ("trimmed seconds from start of voice clip: " + str((original_length - final_len) / 24000.))
        sd.play(voice_clip, samplerate=24000)
        print ("latency to speaking: " + str(round(time.perf_counter() - audio_start_time.value - last_word_time, 2)))
        stop_speaking_event.wait(timeout=len(voice_clip) / 24000.)
        stop_speaking_event.clear()
        sd.stop()

def mic_process(audio_start_time, audio_queue):
    die_if_parent_process_dies_on_linux()
    import pyaudio
    import time
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    while True:
        data = stream.read(1024)
        if not audio_start_time.value:
            audio_start_time.value = time.perf_counter()
        audio_queue.put(data)

def whisper_process(audio_queue, segments, ignore_speech_before, segments_lock):
    die_if_parent_process_dies_on_linux()
    global whisper_model
    from faster_whisper import WhisperModel
    import pyaudio
    import librosa
    import os
    import numpy as np
    import time
    import wave

    for counter in range(1, 2**31):
        wav_filename = f'audio_output_{counter}.wav'
        if not os.path.isfile(wav_filename):
            wav_file = wave.open(wav_filename, 'wb')
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            break
    # TODO: this is pinging Hugging Face via hf_hub something or other, gotta stop it from doing that. discovered when vpn was on and hf refused to respond
    # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16", download_root=None)

    transcription_window = np.zeros([0], dtype=np.float32)
    window_offset = 0
    previous_segments = []
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()
            wav_file.writeframes(data)
            
            transcription_window = np.append(transcription_window, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 16384.0)

        transcription_window_length = 20

        if len(transcription_window) > 16000*transcription_window_length:
            window_offset += (len(transcription_window) - 16000*transcription_window_length)/16000.
            transcription_window = transcription_window[-16000*transcription_window_length:]
        if window_offset < ignore_speech_before.value:
            # increase window_offset to ignore segments before ignore_speech_before
            transcription_window = transcription_window[int((ignore_speech_before.value - window_offset)*16000):]
            window_offset = ignore_speech_before.value
        # TODO: transcribe is a pretty complex (slow?) function, maybe we can skip some of what it does?
        raw_segments, info = whisper_model.transcribe(transcription_window, language="en", beam_size=5, word_timestamps=False, condition_on_previous_text=False)
        
        with segments_lock:
            segments[:] = [(round(segment.start + window_offset, 2), round(segment.end + window_offset, 2), segment.text)
                            for segment in raw_segments if segment.no_speech_prob < 0.15]

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    # Kill all child processes when we die on Windows. On Linux the child process must opt in instead, which is handled by die_if_parent_process_dies_on_linux()
    # TODO figure out macos version of this if we support macos later
    import os
    if os.name == 'nt':
        # This only works on >=Win8, on <=Win7 processes cannot have more than one job object and the shell assigns one to all processes it creates.
        # TODO: do whatever workaround is necessary on Win7 if we care to support it
        # pip install pywin32
        import win32api
        import win32job
        hJob = win32job.CreateJobObject(None, "")
        extended_info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)
        extended_info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation, extended_info)
        win32job.AssignProcessToJobObject(hJob, win32api.GetCurrentProcess())


    audio_start_time = multiprocessing.Value('d', 0)
    audio_queue = multiprocessing.Queue()
    segments = multiprocessing.Manager().list()
    ignore_speech_before = multiprocessing.Value('d', 0)
    segments_lock = multiprocessing.Lock()
    multiprocessing.Process(target=mic_process, args=(audio_start_time, audio_queue)).start()
    multiprocessing.Process(target=whisper_process, args=(audio_queue, segments, ignore_speech_before, segments_lock)).start()

    voice_clips_queue = multiprocessing.Queue()
    stop_speaking_event = multiprocessing.Event()
    multiprocessing.Process(target=play_voice_clips_process, args=(voice_clips_queue, stop_speaking_event, audio_start_time)).start()

    import time
    import os

    from third_party.styletts2 import synthesize
    from exllama.chat import llm, remove_last_prompt

    # TODO figure out how to warm up exllama2
    print ("llm loaded")

    synthesize('Hi.')
    print ("tts initialized")

    # TODO: echo cancellation to filter out our own voice allowing the use of laptop speakers/mic
    # TODO: call out to GPT-4
    # TODO: scale down to local llama 7b and whisper small for smaller gpus. need to fit in 8GB for 50% reach, 12 for 15% reach, 24 has only 2.2% reach, according to steam hardware survey
    # TODO: start speaking first sentence generated by llama while calling gpt-4 and interrupt the llama response with the gpt one when it arrives
    # TODO: when user starts speaking, stop talking
    # TODO: when interrupted by user, remove the unsaid part of the LLM response from the conversation context
    # TODO: find better way to detect if user is still speaking before responding
    # TODO: prompt tuning for various personalities
    # TODO: run in system tray
    # TODO: train a model to classify whether the user wants a response from us or not (e.g. saying um and still thinking, talking to someone else, etc)
    # TODO: package for distribution on both linux and windows
    # TODO: add context to LLM prompt using accessibility APIs to read user's screen
    # TODO: find better TTS, try eleven? want controllable emotion, better prosody, ability to laugh, etc
    # TODO: support multiple langauges at once, for language learning

    from nltk.tokenize import sent_tokenize

    if os.name == 'nt':
        import msvcrt

    to_speak = ''
    llm_generator = None
    last_word_time = 0
    llm_prompt = ''
    print("entering main loop")
    while True:
        next_sentence_to_speak = None
        if llm_generator:
            try:
                to_speak += next(llm_generator)
                sentences = sent_tokenize(to_speak)
                if len(sentences) > 1:
                    next_sentence_to_speak = sentences[0]
                    to_speak = ' '.join(sentences[1:])
            except StopIteration:
                if to_speak:
                    next_sentence_to_speak = to_speak
                    to_speak = ''
                    llm_prompt = ''
                    llm_generator = None
                    with segments_lock:
                        segments[:] = []
                        ignore_speech_before.value = time.perf_counter() - audio_start_time.value
        if next_sentence_to_speak:
            empty = voice_clips_queue.empty()
            if empty:
                print ("speaking: " + next_sentence_to_speak)
            latency = time.perf_counter() - audio_start_time.value - last_word_time
            if voice_clips_queue.empty():
                print ("Latency to LLM response: " + str(round(latency, 2)))
            voice_clips_queue.put((synthesize(next_sentence_to_speak.strip(), speed=1.3), last_word_time))

        with segments_lock:
            # TODO: if a segment disappears this doesn't stop the llm from speaking, it probably should
            if segments:
                print ("got segments: " + str(segments))
                user_spoke = ' '.join([seg[2] for seg in segments])
                # TODO: normalize the prompt for case/whitespace/punctuation/etc when comparing 
                if llm_prompt != user_spoke:
                    if llm_prompt:
                        # TODO: test to see if remove_last_prompt is working
                        # TODO: if a significant part of the response was already spoken, don't remove it from the context
                        print("llm prompt changed, removing last prompt from context")
                        remove_last_prompt()
                    llm_prompt = user_spoke
                    llm_generator = llm(llm_prompt)
                    to_speak = ''
                    last_word_time = segments[-1][1]
                    if not voice_clips_queue.empty():
                        # TODO NEXT: interrupting is not working
                        print ("interrupting because you said " + llm_prompt)
                        while not voice_clips_queue.empty():
                            voice_clips_queue.get_nowait()
                        stop_speaking_event.set()
                        print ("latency to interrupting: " + str(round(time.perf_counter() - audio_start_time.value - last_word_time, 2)))
                    print ("llm prompt: " + llm_prompt)
                    print ("latency to prompting: " + str(round(time.perf_counter() - audio_start_time.value - last_word_time, 2)))

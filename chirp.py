# This is the golden command to copy a build to the test machine fast:
# rclone sync -P --fast-list --size-only C:\src\chirp\dist\chirp \\old-titan\Users\jdarp\Desktop\chirp
# TODO: rename to Chirpy?
# TODO: why does the first response take longer?
# TODO: why does it hang and stop responding to ctrl-c sometimes
# TODO: how can we speed up the voice without artifacts
# TODO: upgrade to multi speaker model from styletts2 when it is released in a few weeks
# TODO: figure out why low_mem arg to exllamav2 crashes, it would save about a gig of memory

import multiprocessing
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL) # exit instantly on ctrl-c instead of throwing KeyboardInterrupt because it's not reliable, many blocking calls fail to be interrupted by KeyboardInterrupt
import os
import sys
running_in_pyinstaller = getattr(sys, 'frozen', False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # suppresses: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['NLTK_DATA'] = os.path.join(sys._MEIPASS, 'nltk_data') if running_in_pyinstaller else "/src/models/nltk_data"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(sys._MEIPASS, 'eSpeak/libespeak-ng.dll') if running_in_pyinstaller else "/src/models/eSpeak/libespeak-ng.dll"
os.environ["ESPEAK_DATA_PATH"] = os.path.join(sys._MEIPASS, 'eSpeak/espeak-ng-data') if running_in_pyinstaller else "/src/models/eSpeak/espeak-ng-data"

# Fix unable to load cublas64_11.dll error on windows
if not running_in_pyinstaller:
    # os.add_dll_directory('C:\\src\\models\\dlls\\') # this actually does not work, sigh
    sys.path.append('C:\\src\\models\\dlls\\') # this actually works

def die_if_parent_process_dies_on_linux():
    try:
        import pyprctl
        pyprctl.set_pdeathsig(signal.SIGKILL)
    except ImportError:
        pass

def play_voice_clips_process(voice_clips_queue, audio_start_time, skip_responses_before, last_response_spoken):
    die_if_parent_process_dies_on_linux()
    import sounddevice as sd
    import time
    import numpy as np
    import librosa
    stream = sd.OutputStream(samplerate=24000, channels=1)
    stream.start()
    librosa.effects.trim(np.zeros([1024]), top_db=40)
    print ("Audio output initialized")
    last_response_index = -1
    while True:
        (voice_clip, text, last_word_time, index) = voice_clips_queue.get()
        if voice_clip is None:
            break
        if index < skip_responses_before.value:
            continue
        # voice_clip = librosa.effects.trim(voice_clip, top_db=20)[0]
        stream.write(voice_clip)
        last_response_spoken.value = index
        if last_word_time and last_response_index != index:
            print (f"latency to speaking response number {index}: {round(time.perf_counter() - audio_start_time.value - last_word_time, 2)}")
            last_response_index = index
    stream.stop()
    stream.close()

def tts_process(tts_queue, voice_clips_queue, skip_responses_before, generating_response_number, tts_ready):
    die_if_parent_process_dies_on_linux()
    import time
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    import numpy as np
    import torch
    best_xtts_speakers = [
        'Tammy Grit',
        'Royston Min',
        'Craig Gutsy',
        'Gilberto Mathias',
        'Barbora MacLean',
        'Wulf Carlevaro',
        'Kumar Dahl',
        'Luis Moray',
        ]
    def synthesize(text, language="en", speaker='Barbora MacLean', speed=1):
        print(f"Synthesizing: {text}")
        gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.speaker_manager.speakers[speaker].values()
        gpt_cond_latent = gpt_cond_latent.to("cuda")
        speaker_embedding = speaker_embedding.to("cuda")
        start_time = time.time()
        chunks = tts.synthesizer.tts_model.inference_stream(text, language, gpt_cond_latent, speaker_embedding, enable_text_splitting=True, stream_chunk_size=7)
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunk: {round(time.time() - start_time, 2)}")
            else:
                print(f"Generated speech in {round((time.time() - start_time) / (len(chunk) / 24000.), 2)}x real time")
            yield chunk.cpu().numpy()
            start_time = time.time()

        # return np.array(tts.tts(text=text, speaker=speaker, language='en'))
    test_text = "How much wood would a woodchuck chuck if a woodchuck could chuck wood? Peter Piper picked a peck of pickled peppers."
    for chunk in synthesize(test_text):
        voice_clips_queue.put((chunk, 0, test_text, -1))

    tts_ready.set()
    print ("tts loaded")

    while True:
        (text, last_word_time, index) = tts_queue.get()
        generating_response_number.value = index
        if index >= skip_responses_before.value:
            for chunk in synthesize(text):
                if index < skip_responses_before.value:
                    break
                voice_clips_queue.put((chunk, text, last_word_time, index))
        generating_response_number.value = -1

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
    i = 0
    while True:
        data = stream.read(1024)
        i += 1024
        # print(f"{stream.get_time()} {time.perf_counter()} {audio_start_time.value + i/16000.}")
        if not audio_start_time.value:
            audio_start_time.value = time.perf_counter()
            print(" ______________________________________ ")
            print("|                                      |")
            print("| Microphone open. Start speaking now! |")
            print("|______________________________________|")
            print("")
        audio_queue.put((data, stream.get_time() - audio_start_time.value))

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

    wav_file = None
    # for counter in range(1, 2**31):
    #     wav_filename = f'audio_output_{counter}.wav'
    #     if not os.path.isfile(wav_filename):
    #         wav_file = wave.open(wav_filename, 'wb')
    #         wav_file.setnchannels(1)
    #         wav_file.setsampwidth(2)
    #         wav_file.setframerate(16000)
    #         break
    whisper_model = WhisperModel("large-v2", device="cuda", compute_type="float16", download_root=sys._MEIPASS if running_in_pyinstaller else """c:\src\models""")

    transcription_window = np.zeros([0], dtype=np.float32)
    raw_segments, info = whisper_model.transcribe(np.ones([16000*5], dtype=np.float32), language="en", beam_size=5, word_timestamps=False, condition_on_previous_text=False)
    [segment for segment in raw_segments] # iterate over the generator to force it to load the model
    print ("Voice recognition loaded")

    window_offset = 0
    previous_segments = []
    while True:
        (data, data_time) = audio_queue.get()
        while data:
            if wav_file: wav_file.writeframes(data)
            transcription_window = np.append(transcription_window, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 16384.0)
            try:
                (data, data_time) = audio_queue.get_nowait()
            except multiprocessing.queues.Empty:
                data = None

        transcription_window_length = 20

        silence_buffer_s = 1 # Whisper doesn't like having a single word right at the beginning of the transcription, so add a little silence before the audio.

        if len(transcription_window) > 16000*transcription_window_length:
            window_offset += (len(transcription_window) - 16000*transcription_window_length)/16000.
            transcription_window = transcription_window[-16000*transcription_window_length:]
        if window_offset < ignore_speech_before.value:
            # increase window_offset to ignore segments before ignore_speech_before
            transcription_window = transcription_window[int((ignore_speech_before.value - window_offset)*16000):]
            window_offset = ignore_speech_before.value
        start_of_window = data_time - len(transcription_window)/16000.
        # TODO: transcribe is a pretty complex (slow?) function, maybe we can skip some of what it does?
        raw_segments, info = whisper_model.transcribe(np.concatenate((np.zeros(silence_buffer_s * 16000), transcription_window)), language="en", beam_size=5, word_timestamps=False, condition_on_previous_text=False)
        raw_segments = [segment for segment in raw_segments]
        # print (raw_segments)
        with segments_lock:
            segments[:] = [(round(start_of_window + segment.start - silence_buffer_s, 2), round(segment.end + start_of_window - silence_buffer_s, 2), segment.text)
                            for segment in raw_segments if segment.no_speech_prob < 0.2]

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
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)

    print ("Loading...")

    audio_start_time = multiprocessing.Value('d', 0)
    audio_queue = multiprocessing.Queue()
    segments = multiprocessing.Manager().list()
    ignore_speech_before = multiprocessing.Value('d', 0)
    segments_lock = multiprocessing.Lock()
    multiprocessing.Process(target=whisper_process, args=(audio_queue, segments, ignore_speech_before, segments_lock)).start()

    voice_clips_queue = multiprocessing.Queue()
    tts_queue = multiprocessing.Queue()
    skip_responses_before = multiprocessing.Value('d', -1)
    generating_response_number = multiprocessing.Value('d', -1)
    last_response_spoken = multiprocessing.Value('d', -1)
    tts_ready = multiprocessing.Event()
    multiprocessing.Process(target=play_voice_clips_process, args=(voice_clips_queue, audio_start_time, skip_responses_before, last_response_spoken)).start()
    multiprocessing.Process(target=tts_process, args=(tts_queue, voice_clips_queue, skip_responses_before, generating_response_number, tts_ready)).start()

    import time
    import os
    import nltk
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 

    try:
        from exllama.chat import llm, remove_last_prompt
    except RuntimeError:
        print ("Failed to load pytorch CUDA. Nvidia GPU required. Try upgrading your Nvidia drivers.")
        exit(1)

    # TODO figure out how to warm up exllama2
    print ("llm loaded")

    # Wait for tts to load
    tts_ready.wait()

    multiprocessing.Process(target=mic_process, args=(audio_start_time, audio_queue)).start()

    # TODO: echo cancellation to filter out our own voice allowing the use of laptop speakers/mic
    # TODO: call out to GPT-4
    # TODO: figure out an 8GB VRAM mode for 50% reach instead of 15% at 12GB
    # TODO: start speaking first sentence generated by llama while calling gpt-4 and interrupt the llama response with the gpt one when it arrives
    # TODO: when interrupted by user, remove the unsaid part of the LLM response from the conversation context
    # TODO: find better way to detect if user is still speaking before responding
    # TODO: prompt tuning for various personalities
    # TODO: run in system tray
    # TODO: train a model to classify whether the user wants a response from us or not (e.g. saying um and still thinking, talking to someone else, etc)
    # TODO: package for distribution on both linux and windows
    # TODO: add context to LLM prompt using accessibility APIs to read user's screen
    # TODO: support multiple langauges at once, for language learning

    if os.name == 'nt':
        import msvcrt

    to_speak = ''
    llm_generator = None
    sentences_spoken = 0
    last_word_time = 0
    llm_prompt = ''
    normalized_llm_prompt = ''
    last_debug_print = ''
    response_number = 0

    while True:
        next_sentence_to_speak = None
        if llm_generator:
            try:
                to_speak += next(llm_generator)
                sentences = [sentence for line in to_speak.splitlines() for sentence in nltk.sent_tokenize(line)]
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
            sentences_spoken += 1
            # print ("speaking: " + next_sentence_to_speak)
            latency = time.perf_counter() - audio_start_time.value - last_word_time
            response_interval_start = 0
            if sentences_spoken == 1:
                print ("Latency to LLM response: " + str(round(latency, 2)))
                response_interval_start = last_word_time
            tts_queue.put((next_sentence_to_speak.strip(), response_interval_start, response_number))

        with segments_lock:
            # TODO: if a segment disappears this doesn't stop the llm from speaking, it probably should
            if segments:
                user_spoke = ' '.join([seg[2].strip() for seg in segments])
                def normalize(str):
                    return ' '.join(nltk.word_tokenize(str.lower()))
                normalized_user_spoke = normalize(user_spoke)
                # If the prompt changed after normalization, and the user actually spoke recently (so this isn't just a delayed response to something they said a while ago), then prompt the LLM
                if normalized_llm_prompt != normalized_user_spoke and segments:
                    print ("user spoke: " + str(user_spoke))
                    speech_end_time = segments[-1][1]
                    current_time = time.perf_counter() - audio_start_time.value
                    if speech_end_time > current_time + 0.2:
                        pass # print ("future speech, ignoring. last word time: " + str(segments[-1][1]) + " time: " + str(time.perf_counter() - audio_start_time.value))
                    elif segments[-1][1] > time.perf_counter() - audio_start_time.value - 2:
                        # print ("user spoke recently, prompting LLM. last word time: " + str(segments[-1][1]) + " time: " + str(time.perf_counter() - audio_start_time.value))
                        if llm_prompt:
                            # TODO: test to see if remove_last_prompt is entirely working 
                            # TODO: if a significant part of the response was already spoken, don't remove it from the context. ideally detect how many words have been spoken and only remove the ones that haven't been spoken yet, appending "..." to indicate truncation to the llm
                            print("llm prompt changed, removing last prompt from context")
                            remove_last_prompt()
                        llm_prompt = user_spoke
                        normalized_llm_prompt = normalized_user_spoke
                        llm_generator = llm(llm_prompt)
                        to_speak = ''
                        last_word_time = segments[-1][1]
                        sentences_spoken = 0
                        response_number += 1
                        if not voice_clips_queue.empty() or generating_response_number.value >= 0:
                            print (f"interrupting voice clips before {response_number} because you said " + llm_prompt)
                            print ("latency to interrupting: " + str(round(time.perf_counter() - audio_start_time.value - last_word_time, 2)))
                        skip_responses_before.value = response_number
                        print ("latency to prompting: " + str(round(time.perf_counter() - audio_start_time.value - last_word_time, 2)))
                    # else:
                    #     print ("user spoke a while ago, ignoring. last word time: " + str(segments[-1][1]) + " time: " + str(time.perf_counter() - audio_start_time.value))

# TODO NEXT!!! move whisper to own process, performance sucks with GIL too much for same process

def play_voice_clips(voice_clips_queue, wake_up_event):
    import sounddevice as sd
    import numpy as np
    import time
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
        print ("latency to speaking: " + str(round(time.perf_counter() - last_word_time, 2)))
        wake_up_event.wait(timeout=len(voice_clip) / 24000.)
        wake_up_event.clear()
        sd.stop()

import multiprocessing
import socket
import pyaudio
import queue
import wave

def run_whisper(audio_start_time, segments, ignore_segments_before, segments_lock):
    global whisper_model
    from faster_whisper import WhisperModel
    import librosa
    import os
    import numpy as np
    import time

    import os
    wav_filename = 'audio_output_1.wav'
    counter = 1
    while os.path.isfile(wav_filename):
        counter += 1
        wav_filename = f'audio_output_{counter}.wav'


    # TODO: this is pinging Hugging Face via hf_hub something or other, gotta stop it from doing that. discovered when vpn was on and hf refused to respond
    # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    whisper_model = WhisperModel("large-v2", device="cuda", compute_type="float16", download_root=None)
    print ('loaded whisper model')


    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    wav_file = wave.open(wav_filename, 'wb')
    wav_file.setnchannels(CHANNELS)
    wav_file.setsampwidth(p.get_sample_size(FORMAT))
    wav_file.setframerate(RATE)

    transcription_window = np.zeros([0], dtype=np.float32)
    window_offset = 0
    previous_segments = []
    while True:
        while True:
            data = stream.read(CHUNK)
            if not audio_start_time.value:
                audio_start_time.value = time.perf_counter()
            # wav_file.writeframes(data)
            
            transcription_window = np.append(transcription_window, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 16384.0)
            # print (stream.get_read_available())
            if stream.get_read_available() < CHUNK:
                break

        def get_overlapping_segments(new_segment, segments, overlap_threshold=0.1):
            overlapping_segments = []
            for i, segment in enumerate(segments):
                overlap = min(new_segment[1], segment[1]) - max(new_segment[0], segment[0])
                if overlap > overlap_threshold * (new_segment[1] - new_segment[0]):
                    overlapping_segments.append(i)
            return overlapping_segments

        transcription_window_length = 20

        if len(transcription_window) > 16000*transcription_window_length:
            window_offset += (len(transcription_window) - 16000*transcription_window_length)/16000.
            transcription_window = transcription_window[-16000*transcription_window_length:]
        if window_offset < ignore_segments_before.value:
            # increase window_offset to ignore segments before ignore_segments_before
            transcription_window = transcription_window[int((ignore_segments_before.value - window_offset)*16000):]
            window_offset = ignore_segments_before.value
        # TODO: transcribe is a pretty complex (slow?) function, maybe we can skip some of what it does?
        raw_segments, info = whisper_model.transcribe(transcription_window, language="en", beam_size=5, word_timestamps=False, condition_on_previous_text=False)
        
        with segments_lock:
            new_segments = []
            end_of_window = len(transcription_window)/16000.
            for segment in raw_segments:
                # print ("got segment: " + segment.text)
                if segment.no_speech_prob < 0.2:
                    new_segment = (segment.start + window_offset, segment.end + window_offset, segment.text)
                    if get_overlapping_segments(new_segment, new_segments):
                        print ("ERROR OVERLAPPING SEGMENTS IN TRANSCRIPTION")
                    new_segments.append(new_segment)
            segments[:] = []
            segments.extend(new_segments)
            # segments = list(filter(lambda x: x[1] < window_offset + transcription_window_length/2, list(segments)))
            # segments.extend(filter(lambda x: x[1] >= window_offset + transcription_window_length/2, new_segments))
            # segments.sort(key=lambda x: x[0])
            # print(segments)
            # Remove segments that overlap by more than 0.2 seconds
            # i = 0
            # while i < len(segments) - 1:
            #     if segments[i+1][0] - segments[i][1] < 0.2:
            #         segments.pop(i+1)
            #     else:
            #         i += 1

            joined_text = ' '.join([segment[2] for segment in list(segments)])
            # print(joined_text)
            

            # segments.filter(lambda x: x[1] > window_offset + end_of_window/2)
            # segments.extend(new_segments)
            # for segment in new_segments:
            #     overlapping_segments = get_overlapping_segments(segment, previous_segments)
            #     for i in overlapping_segments:
            #         segments.pop(i)
            #     if not overlapping_segments:
            #         segments.append(segment)
            # TODO: also remove segments that should have appeared in the new transcription but didn't
            # segments.sort(key=lambda x: x[0])

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    audio_start_time = multiprocessing.Value('d', 0)

    segments = multiprocessing.Manager().list()
    ignore_segments_before = multiprocessing.Value('d', 0)
    segments_lock = multiprocessing.Lock()

    whisper_process = multiprocessing.Process(target=run_whisper, args=(audio_start_time, segments, ignore_segments_before, segments_lock), daemon=True)
    whisper_process.start()


    # # part of workaround for torchvision pyinstaller interaction bug from https://github.com/pytorch/vision/issues/1899
    # def script_method(fn, _rcb=None):
    #     return fn
    # def script(obj, optimize=True, _frames_up=0, _rcb=None):
    #     return obj    
    # import torch.jit
    # torch.jit.script_method = script_method 
    # torch.jit.script = script

    import time
    from timeit import default_timer as timer
    import subprocess
    import signal
    import select
    import sys
    import os
    import json
    import numpy as np
    import sounddevice as sd
    # sd.default.samplerate = 24000
    # sd.default.channels = 1
    # sd.default.latency = 'low'

    try:
        import pyprctl
        preexec_fn = lambda : pyprctl.set_pdeathsig(signal.SIGKILL)
    except ImportError:
        preexec_fn = None

    if os.name == 'nt':
        # This only works on >=Win8, on <=Win7 processes cannot have more than one job object and the shell assigns one to all processes it creates.
        # TODO: do whatever workaround is necessary on Win7 if we care to support it
        # pip install pywin32
        import win32api
        import win32job
        # hJob will be GC'd when our process exits and this will kill all children created with CreateProcess, hopefully no dependencies create processes in other sneaky ways
        hJob = win32job.CreateJobObject(None, "")
        extended_info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)
        extended_info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation, extended_info)
        win32job.AssignProcessToJobObject(hJob, win32api.GetCurrentProcess())



    from third_party.styletts2 import synthesize
    from exllama.chat import llm, remove_last_prompt


    # from TTS.api import TTS
    # tts = TTS(model_name='tts_models/en/jenny/jenny', gpu=True,)
    # print(tts.speakers)
    # wav = synthesize("This is a test? This is also a test!!", speed=1.2)
    # sd.play(wav, samplerate=24000)
    # sd.wait()

    # only works on Linux
    # arecord = subprocess.Popen(["arecord", "-f", "S16_LE", "-c1", "-r", "16000", "-t", "raw", "-D", "default"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, preexec_fn=preexec_fn)
    # nc = subprocess.Popen(["nc", "localhost", "43007"], stdin=arecord.stdout, stdout=subprocess.PIPE, preexec_fn=preexec_fn)

    # mlc_llm = subprocess.Popen(["../mlc-llm/build/mlc_chat_cli", "--local-id", "Llama-2-13b-chat-hf-q4f16_1"], cwd="../mlc-llm/", stdin=subprocess.PIPE, stdout=subprocess.PIPE, preexec_fn=preexec_fn)

    synthesize('Hi.')
    # sd.play(np.zeros(0), samplerate=24000)
    print ("tts initialized")

    # time.sleep(100)

    # Wait until mlc_llm finishes loading
    # while True:
    #     output = mlc_llm.stdout.read1()
    #     if output is not None:
    #         output = output.decode()
    #         if "[INST]:" in output:
    #             break
    # for token in llm("Hi."):
    #     print(token, end=' ')

    # print ("llm initialized")

    # TODO figure out how to warm up exllama2


    # TODO: echo cancellation to filter out our own voice allowing the use of laptop speakers/mic
    # TODO: find a better way of detecting when the user is done speaking, period at end of sentence is not reliable with whisper plus VAD. alternatively fix VAD using whisper itself.
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
    # TODO: filter easy whisper hallucinations like "Thanks for watching!"
    # TODO: find better TTS, try eleven? want controllable emotion, higher speed, ability to laugh, etc
    # TODO: support multiple langauges at once, for language learning
    # TODO: delete whisper's repeated words after a long pause

    import multiprocessing

    voice_clips_queue = multiprocessing.Queue()
    wake_up_event = multiprocessing.Event()

    voice_clips_process = multiprocessing.Process(target=play_voice_clips, args=(voice_clips_queue, wake_up_event), daemon=True)
    voice_clips_process.start()

    from nltk.tokenize import sent_tokenize

    if os.name == 'nt':
        import msvcrt

    accumulated_input = ""
    to_speak = ''
    llm_generator = None
    print("entering main loop")
    last_word_time = 0
    llm_prompt = ''
    while True:
        time.sleep(0)
        output = None
        # if msvcrt:
        #     if msvcrt.kbhit():
        #         output = os.read(sys.stdin.fileno(), 999999).decode().replace('\n', ' ').strip()
        #         last_word_time = time.perf_counter()
        #         segments.append((last_word_time, last_word_time, output))
        
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
                        ignore_segments_before.value = time.perf_counter()
        if next_sentence_to_speak:
            empty = voice_clips_queue.empty()
            if empty:
                print ("speaking: " + next_sentence_to_speak)
            latency = time.perf_counter() - last_word_time
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
                        # TODO NEXT: THIS IS SCREWING UP! causes multiple responses to be jumbled together I think
                        # TODO: if a significant part of the response was already spoken, don't remove it from the context
                        print("llm prompt changed, removing last prompt from context")
                        remove_last_prompt()
                    llm_prompt = user_spoke
                    llm_generator = llm(llm_prompt)
                    last_word_time = segments[-1][1]
                    if not voice_clips_queue.empty():
                        print ("interrupting because you said " + llm_prompt)
                        while not voice_clips_queue.empty():
                            voice_clips_queue.get_nowait()
                    print ("llm prompt: " + llm_prompt)

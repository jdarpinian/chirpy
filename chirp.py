# TODO NEXT!!! put audio playback in separate process to fix buffer underruns, hopefully?
# TODO NEXT!!! record dataset of voice conversations to optimize whisper_streaming latency and hallucinations, create tools for visualization

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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')


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



    # TODO: this is pinging Hugging Face via hf_hub something or other, gotta stop it from doing that. discovered when vpn was on and hf refused to respond
    # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    whisper_online_server = subprocess.Popen(["python", "third_party/whisper_streaming/whisper_online_server.py", "--model", "large-v2", "--min-chunk-size", "0.2"], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, preexec_fn=preexec_fn) # "--vad", 


    from third_party.styletts2 import synthesize
    from exllama.chat import llm


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

    # Wait until whisper_online_server outputs "Listening" to stderr
    while True:
        output = whisper_online_server.stderr.readline()
        if output is not None:
            encoding = 'utf-8'
            if os.name == 'nt':
                encoding = 'latin-1'
            output = output.decode(encoding=encoding)
            if "Listening" in output:
                break

    import threading
    import socket
    import pyaudio
    import queue
    import wave

    audio_start_time = 0
    import os
    wav_filename = 'audio_output_1.wav'
    counter = 1
    while os.path.isfile(wav_filename):
        counter += 1
        wav_filename = f'audio_output_{counter}.wav'

    def stream_audio_to_server(audio_queue):
        global audio_start_time
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 43007))
        s.setblocking(0)

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        wav_file = wave.open(wav_filename, 'wb')
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(p.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)

        while True:
            data = stream.read(CHUNK)
            if not audio_start_time:
                audio_start_time = time.perf_counter()

            s.sendall(data)
            wav_file.writeframes(data)

            try:
                chunk = s.recv(CHUNK)
                if chunk:
                    chunk = chunk.strip(b'\x00')
                    chunk = chunk.decode()
                    chunk = chunk.strip()
                    if chunk:
                        audio_queue.put(chunk)
                    chunk = None
            except socket.error:
                pass

    audio_queue = queue.Queue()
    audio_thread = threading.Thread(target=stream_audio_to_server, args=(audio_queue,), daemon=True)
    audio_thread.start()

    print ("whisper initialized")

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


    import re
    thanks_for_watching = re.compile(r"(thanks for watching[.!]?)", re.IGNORECASE)

    import queue
    import threading

    import multiprocessing

    voice_clips_queue = multiprocessing.Queue()
    wake_up_event = multiprocessing.Event()

    voice_clips_process = multiprocessing.Process(target=play_voice_clips, args=(voice_clips_queue, wake_up_event), daemon=True)
    voice_clips_process.start()

    def read_stderr():
        while True:
            err = whisper_online_server.stderr.readline()
            if err == '' and whisper_online_server.poll() is not None:
                break
            # if err:
            #     print(err.strip())

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()



    from nltk.tokenize import sent_tokenize

    if os.name == 'nt':
        import msvcrt

    accumulated_input = ""
    to_speak = ''
    llm_generator = iter(())
    print("entering main loop")
    last_word_time = 0
    while True:
        output = None
        if msvcrt:
            if msvcrt.kbhit():
                output = os.read(sys.stdin.fileno(), 999999).decode().replace('\n', ' ').strip()
                last_word_time = time.perf_counter()
        elif select.select([sys.stdin,],[],[],0.0)[0]:
            output = os.read(sys.stdin.fileno(), 999999).decode().replace('\n', ' ').strip()
            last_word_time = time.perf_counter()
        
        if output is None and not audio_queue.empty():
            output = audio_queue.get_nowait()
            output = output.replace('\n', ' ').strip()
            words = output.split(' ')
            timing = words[:2]
            last_word_time = audio_start_time + float(timing[1])/1000.
            latency = time.perf_counter() - last_word_time
            print ("ASR latency: " + str(round(latency, 2)))
            words = words[2:]
            output = ' '.join([i for i in words if i]).strip()
        
        next_sentence_to_speak = None
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
        if next_sentence_to_speak:
            empty = voice_clips_queue.empty()
            if empty:
                print ("speaking: " + next_sentence_to_speak)
            latency = time.perf_counter() - last_word_time
            if voice_clips_queue.empty():
                print ("Latency to LLM response: " + str(round(latency, 2)))
            voice_clips_queue.put((synthesize(next_sentence_to_speak.strip(), speed=1.3), last_word_time))

        if output is not None:
            accumulated_input += output + ' '
            print ("accumulated input: " + accumulated_input)
            accumulated_input = thanks_for_watching.sub("", accumulated_input)

            if (len(accumulated_input.strip()) > 3 and not voice_clips_queue.empty()):
                print ("interrupting because you said " + accumulated_input.strip())
                # TODO: remove unsaid part of response from context, this is actually really important
                while not voice_clips_queue.empty():
                    voice_clips_queue.get_nowait()
                wake_up_event.set()

            if accumulated_input.strip().endswith(('.', '?', '!')) and not accumulated_input.strip().endswith('...') and len(accumulated_input.strip()) > 3:
                latency = time.perf_counter() - last_word_time
                print ("Latency to LLM init: " + str(round(latency, 2)))
                llm_generator = llm(accumulated_input.strip())
                accumulated_input = ''
            #     mlc_llm.stdin.write(accumulated_input.strip().encode() + b'\n')
            #     mlc_llm.stdin.flush()
            #     to_speak = ""
            #     while True:
            #         read = mlc_llm.stdout.read1().decode()
            #         to_speak += read.replace("[/INST]:", "")
            #         if "[INST]:" in read:
            #             # Done generating text, speak it all.
            #             to_speak = to_speak.replace("[INST]:", '').strip()
            #             print (to_speak)
            #             voice_clips_queue.put(synthesize(to_speak))
            #             break
            #         # If we've generated a full sentence, send it to TTS right away before the rest of the response is generated.
            #         sentences = tts.synthesizer.split_into_sentences(to_speak)
            #         if (len(sentences) > 1):
            #             to_speak = sentences[-1]
            #             for sentence in sentences[:-1]:
            #                 print (sentence)
            #                 wav = synthesize(sentence.strip())
            #                 voice_clips_queue.put(wav)
            #     accumulated_input = ''

#!/usr/bin/env python
# coding: utf-8

# # StyleTTS 2 Demo (LJSpeech)
# 

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# TODO: package espeak with pyinstaller build
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = 'C:\\Program Files\\eSpeak NG\\libespeak-ng.dll'

# TODO: package this download with pyinstaller build, nltk saves it in C:\Users\jdarp\AppData\Roaming\nltk_data...
import nltk
nltk.download('punkt')

# ### Utils

# In[3]:


import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)


# In[4]:


# In[5]:


# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize


# wow python's module system is shit, no way to do relative imports that works in any sensible way, so this is how we import things not in our current directory
import sys
styletts2_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "StyleTTS2/")
sys.path.append(styletts2_path)
saved_cwd = os.curdir
os.chdir(styletts2_path)

from .StyleTTS2.models import *
from .StyleTTS2.utils import *
from .StyleTTS2.text_utils import TextCleaner
textclenaer = TextCleaner()



# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[7]:


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)
    
    return reference_embeddings


# ### Load models

# In[11]:
# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)


# In[ ]:


config = yaml.safe_load(open('Models/LJSpeech/config.yml'))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from .StyleTTS2.Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)


# In[ ]:


model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]


# In[ ]:


params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
params = params_whole['net']


# In[ ]:


for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]


# In[ ]:


from .StyleTTS2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


# In[ ]:


sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)


# ### Synthesize speech

# In[ ]:


# synthesize a text
text = ''' StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis. '''
text = ''' What is the airspeed velocity of an unladen swallow? '''


# In[ ]:


def inference(text, noise, diffusion_steps=5, embedding_scale=1, speed=1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise, 
              embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
              embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()/speed).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()

def synthesize(text, speed=1):
    noise = torch.randn(1,1,256).to(device)
    return inference(text, noise, diffusion_steps=5, embedding_scale=1, speed=speed)


# # #### Basic synthesis (5 diffusion steps)

# # In[ ]:
# import sounddevice as sd


# start = time.time()
# noise = torch.randn(1,1,256).to(device)
# wav = inference(text, noise, diffusion_steps=5, embedding_scale=1)
# rtf = (time.time() - start) / (len(wav) / 24000)
# print(f"RTF = {rtf:5f}")
# sd.play(wav, samplerate=24000)
# sd.wait()


# # #### With higher diffusion steps (more diverse)
# # Since the sampler is ancestral, the higher the stpes, the more diverse the samples are, with the cost of slower synthesis speed.

# # In[ ]:


# start = time.time()
# noise = torch.randn(1,1,256).to(device)
# wav = inference(text, noise, diffusion_steps=100, embedding_scale=2)
# rtf = (time.time() - start) / (len(wav) / 24000)
# print(f"RTF = {rtf:5f}")
# sd.play(wav, samplerate=24000)
# sd.wait()

# # ### Speech expressiveness
# # The following section recreates the samples shown in [Section 6](https://styletts2.github.io/#emo) of the demo page.

# # #### With embedding_scale=1
# # This is the classifier-free guidance scale. The higher the scale, the more conditional the style is to the input text and hence more emotional. 

# # In[ ]:


# texts = {}
# texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
# texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
# texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
# texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

# for k,v in texts.items():
#     noise = torch.randn(1,1,256).to(device)
#     wav = inference(v, noise, diffusion_steps=10, embedding_scale=1)
#     print(k + ": ")
#     sd.play(wav, samplerate=24000)
#     sd.wait()


# # #### With embedding_scale=2

# # In[ ]:


# texts = {}
# texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
# texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
# texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
# texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

# for k,v in texts.items():
#     noise = torch.randn(1,1,256).to(device)
#     wav = inference(v, noise, diffusion_steps=10, embedding_scale=2) # embedding_scale=2 for more pronounced emotion
#     print(k + ": ")
#     sd.play(wav, samplerate=24000)
#     sd.wait()


# # ### Long-form generation
# # This section includes basic implementation of Algorithm 1 in the paper for consistent longform audio generation. The example passage is taken from [Section 5](https://styletts2.github.io/#long) of the demo page. 

# # In[ ]:


# passage = '''If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. Yet for first-class homemade products there is a market in all large cities. All first-class grocers have customers who purchase such goods.'''


# # In[ ]:


# def LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
#     text = text.strip()
#     text = text.replace('"', '')
#     ps = global_phonemizer.phonemize([text])
#     ps = word_tokenize(ps[0])
#     ps = ' '.join(ps)

#     tokens = textclenaer(ps)
#     tokens.insert(0, 0)
#     tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
#     with torch.no_grad():
#         input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
#         text_mask = length_to_mask(input_lengths).to(tokens.device)

#         t_en = model.text_encoder(tokens, input_lengths, text_mask)
#         bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
#         d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

#         s_pred = sampler(noise, 
#               embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
#               embedding_scale=embedding_scale).squeeze(0)
        
#         if s_prev is not None:
#             # convex combination of previous and current style
#             s_pred = alpha * s_prev + (1 - alpha) * s_pred
        
#         s = s_pred[:, 128:]
#         ref = s_pred[:, :128]

#         d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

#         x, _ = model.predictor.lstm(d)
#         duration = model.predictor.duration_proj(x)
#         duration = torch.sigmoid(duration).sum(axis=-1)
#         pred_dur = torch.round(duration.squeeze()).clamp(min=1)

#         pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
#         c_frame = 0
#         for i in range(pred_aln_trg.size(0)):
#             pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
#             c_frame += int(pred_dur[i].data)

#         # encode prosody
#         en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
#         F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
#         out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
#                                 F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
#     return out.squeeze().cpu().numpy(), s_pred


# # In[ ]:


# sentences = passage.split('.') # simple split by comma
# wavs = []
# s_prev = None
# for text in sentences:
#     if text.strip() == "": continue
#     text += '.' # add it back
#     noise = torch.randn(1,1,256).to(device)
#     wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
#     wavs.append(wav)
# # display(ipd.Audio(np.concatenate(wavs), rate=24000, normalize=False))

os.chdir(saved_cwd)
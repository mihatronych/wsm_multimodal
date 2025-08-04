import time
from datetime import datetime, timedelta
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import soundfile as sf
import whisper_timestamped as whisper

import os
import sys
import torch
from .contradictions_list import get_norm_text

model = whisper.load_model("openai/whisper-large-v3-turbo", device="cuda")

def asr_from_aud_seg(ds, ds_type="devel", disease="depression", sr=16000, modality="aud"):
    i = 0
    for k, v_id in enumerate(list(ds["video_id"])):
        i += 1
        print(v_id, str(i))
        path = f"./{disease}/{ds_type}_labels/{v_id}"
        if glob(f'{path}/segments/*.wav'):
            auds = glob(f'{path}/segments/*.wav')
            for n, aud_path in enumerate(auds):
                try:
                    a = whisper.load_audio(aud_path)
                    lines = []
                    sentences = []
                    print("ASR STARTED")
                    start_t = time.time()
                    result = whisper.transcribe(model, a, language="en", detect_disfluencies=True, remove_empty_words=True, condition_on_previous_text=True, remove_punctuation_from_words=True, vad=True)
                    print("ASR ENDED")
                    print("--- %s seconds ---" % (time.time() - start_t))
                    if clean:
                        print("CLEAN DUPLICATES")
                        result = clean(result)
                        print("CLEANED DUPLICATES")
                    text = result["text"]
                    path_to_txt = f"{str.replace(aud_path, ".wav", ".txt")}"
                    with open(path_to_txt, "w", encoding="utf-8") as text_file:
                        text_file.write(get_norm_text(text))

                    for segment in result["segments"]:
                        sentence = {}
                        words = []
                        if segment["words"]:
                            for word in segment["words"]:
                                w = {}
                                if get_norm_text(word["text"]) != "":
                                    w["word"] = get_norm_text(word["text"])
                                    w["start_time"] = word["start"]
                                    w["duration"] = round(word["end"] - word["start"], 6)
                                    words.append(w)
                            sentence["words"] = words
                            sentence["confidence"] = segment["confidence"]
                            sentences.append(sentence)

                    path_to_json = str.replace(aud_path, ".wav", ".json")
                    with open(path_to_json, "w", encoding="utf-8") as final:
                        json.dump(sentences, final)
                    print("TEXT AND JSON ARE WRITTEN")
                except KeyboardInterrupt:
                    return
                except:
                    continue
    return 
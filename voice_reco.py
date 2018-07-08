# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 23:02:47 2018

@author: NitinKhanna
"""

import speech_recognition as sr
sr.__version__
recogniser_obj=sr.Recognizer()

audio_file=sr.AudioFile('president-is-moron.wav')
with audio_file as source:
    audio=recogniser_obj.record(source)

type(audio)


recogniser_obj.recognize_google(audio)

mic = sr.Microphone()
with mic as source:
    recogniser_obj.adjust_for_ambient_noise(source)
    audio=recogniser_obj.listen(source)

recogniser_obj.recognize_google(audio)    
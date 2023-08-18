#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:44:16 2023

@author: jake
"""

import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import soundfile as sf
import base64
import numpy as np

from inference import generate



def main():
    # Streamlit app title
    st.title("Drum Sound Generator")

    # Buttons to select drum type
    condition = st.radio("Select Drum Type:", ["Kick", "Snare", "Hat"])

    # Convert condition to corresponding value
    condition_value = {"Kick": 0, "Snare": 1, "Hat": 2}.get(condition, 0)

    # Generate and plot waveform when button is clicked
    if st.button("Generate Sound"):
        waveform = generate(condition_value).numpy()
        plt.figure(figsize=(8, 4))
        plt.plot(waveform)
        plt.title(f"{condition} Waveform")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        st.pyplot(plt)

        # Play generated audio
        audio_bytes = BytesIO()
        sf.write(audio_bytes, waveform, 16000, format="wav")
        st.audio(audio_bytes, format="audio/wav", start_time=0)

        # Provide a download link for the generated audio
        download_filename = f"{condition}.wav"
        st.download_button("Download Audio", audio_bytes.getvalue(), file_name=download_filename)

if __name__ == '__main__':
    main()


    
# def main():
#     # Streamlit app title
#     st.title("Drum Sound Generator")

#     # Buttons to select drum type
#     condition = st.radio("Select Drum Type:", ["Kick", "Snare", "Hat"])

#     # Convert condition to corresponding value
#     condition_value = {"Kick": 0, "Snare": 1, "Hat": 2}.get(condition, 0)

#     # Generate and plot waveform when button is clicked
#     if st.button("Generate Sound"):
#         waveform = generate(condition_value)
#         plt.figure(figsize=(8, 4))
#         plt.plot(waveform)
#         plt.title(f"{condition} Waveform")
#         plt.xlabel("Time")
#         plt.ylabel("Amplitude")
#         st.pyplot(plt)

#         # Play generated audio
#         audio_bytes = BytesIO()
#         sf.write(audio_bytes, waveform, 16000, format="wav")
#         st.audio(audio_bytes, format="audio/wav", start_time=0)

# if __name__ == '__main__':
#     main()
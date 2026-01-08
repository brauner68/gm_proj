import torch
import soundfile as sf
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from src.vocoder import *


def main():

    path = './low_string.wav'
    vocoder = BigVGAN_Vocoder()
    mel = vocoder.encode(path, T_target=320)
    reconstructed = vocoder.decode(mel)
    vocoder.save_audio(reconstructed.squeeze(), f'./recon_{path[2:-4]}.wav')

    print(f'mel.shape: {mel.shape}')
    print(f'reconstructed.shape: {reconstructed.shape}')

if __name__ == '__main__':
    main()
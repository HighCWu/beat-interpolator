import librosa
import numpy as np
import gradio as gr
import soundfile as sf

from moviepy.editor import *


cache_wav_path = [f'/tmp/{str(i).zfill(2)}.wav' for i in range(50)]
wave_path_iter = iter(cache_wav_path)
cache_mp4_path = [f'/tmp/{str(i).zfill(2)}.mp4' for i in range(50)]
path_iter = iter(cache_mp4_path)

def merge_times(times, times2):
    ids = np.unique(np.where(abs(times2[...,None] - times[None]) < 0.2)[1])
    mask = np.ones_like(times, dtype=np.bool)
    mask[ids] = False
    times = times[mask]
    times = np.concatenate([times, times2])
    times = np.sort(times)

    return times


def beat_interpolator(wave_path, generator, latent_dim, seed, fps=30, batch_size=1, strength=1, max_duration=None, use_peak=False):
    fps = max(10, fps)
    strength = np.clip(strength, 0, 1)
    hop_length = 512
    y, sr = librosa.load(wave_path, sr=24000)
    duration = librosa.get_duration(y=y, sr=sr)

    if max_duration is not None:
        y_len = y.shape[0]
        y_idx = int(y_len * max_duration / duration)
        y = y[:y_idx]

        global wave_path_iter
        try:
            wave_path = next(wave_path_iter)
        except:
            wave_path_iter = iter(cache_wav_path)
            wave_path = next(wave_path_iter)
        sf.write(wave_path, y, sr, subtype='PCM_24')
        y, sr = librosa.load(wave_path, sr=24000)
        duration = librosa.get_duration(y=y, sr=sr)
    
    S = np.abs(librosa.stft(y))
    db = librosa.power_to_db(S**2, ref=np.median).max(0)
    db_mean = np.mean(db)
    db_max = np.max(db)
    db_min = np.min(db)
    db_times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop_length)
    rng = np.random.RandomState(seed)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, hop_length=512, units='time')
    times = np.asarray(beats)
    if use_peak:
        peaks = librosa.util.peak_pick(onset_env, 1, 1, 1, 1, 0.8, 5)
        times2 = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=512)[peaks]
        times2 = np.asarray(times)
        times = merge_times(times, times2)
        
    times = np.concatenate([np.asarray([0.]), times], 0)
    times = list(np.unique(np.int64(np.floor(times * fps / 2))) * 2)

    latents = []
    time0 = 0
    latent0 = rng.randn(latent_dim)
    for time1 in times:
        latent1 = rng.randn(latent_dim)
        db_cur_index = np.argmin(np.abs(db_times - time1.astype('float32') / fps))
        db_cur = db[db_cur_index]
        if db_cur < db_min + (db_mean - db_min) / 3:
            latent1 = latent0 * 0.8 + latent1 * 0.2
        elif db_cur < db_min + 2 * (db_mean - db_min) / 3:
            latent1 = latent0 * 0.6 + latent1 * 0.4
        elif db_cur < db_mean + (db_max - db_mean) / 3:
            latent1 = latent0 * 0.4 + latent1 * 0.6
        elif db_cur < db_mean + 2 * (db_max - db_mean) / 3:
            latent1 = latent0 * 0.2 + latent1 * 0.8
        else:
            pass
        if time1 > duration * fps:
            time1 = int(duration * fps)
        t1 = time1 - time0
        alpha = 0.5 * strength
        latent2 = latent0 * alpha + latent1 * (1 - alpha)
        for j in range(t1):
            alpha = j / t1
            latent = latent0 * (1 - alpha) + latent2 * alpha
            latents.append(latent)
        
        time0 = time1
        latent0 = latent1
        
    outs = []
    ix = 0
    while True:
        if ix + batch_size <= len(latents):
            outs += generator(latents[ix:ix+batch_size])
        elif ix < len(latents):
            outs += generator(latents[ix:])
            break
        else:
            break
        ix += batch_size

    global path_iter
    try:
        video_path = next(path_iter)
    except:
        path_iter = iter(cache_mp4_path)
        video_path = next(path_iter)
    
    video = ImageSequenceClip(outs, fps=fps)
    audioclip = AudioFileClip(wave_path)

    video = video.set_audio(audioclip)
    video.write_videofile(video_path, fps=fps)

    return video_path

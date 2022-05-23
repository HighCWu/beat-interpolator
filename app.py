#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import glob
import pickle
import sys
import importlib
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn

from beat_interpolator import beat_interpolator


def build_models():
    modules = glob.glob('examples/models/*')
    modules = [
        getattr(
            importlib.import_module(
                module.replace('/', '.'),
                package=None
            ),
            'create'
        )
        for module in modules
    ]

    attrs = { module['name']: module  for module in modules }

    return attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def main():
    args = parse_args()
    enable_queue = args.enable_queue
    model_attrs = build_models()

    with gr.Blocks(theme=args.theme) as demo:
        gr.Markdown('''<center><h1>Beat-Interpolator</h1></center>
Play DL models with music beats.<br />
This is a Gradio Blocks app of <a href="https://github.com/HighCWu/beat-interpolator">HighCWu/beat-interpolator</a>.
''')
        with gr.Tabs():
            for model_attr in model_attrs:
                with gr.TabItem(model_attr['name']):
                    generator = model_attr['generator']
                    latent_dim = model_attr['latent_dim']
                    default_fps = model_attr['fps']
                    max_fps = model_attr['fps'] if enable_queue else 60
                    batch_size = model_attr['batch_size']
                    strength = model_attr['strength']
                    default_max_duration  = model_attr['max_duration']
                    max_duration = model_attr['max_duration'] if enable_queue else 360
                    use_peak = model_attr['use_peak']

                    def interpolate(
                        wave_path, 
                        seed, 
                        fps=default_fps, 
                        strength=strength, 
                        max_duration=default_max_duration, 
                        use_peak=use_peak):
                        return beat_interpolator(
                            wave_path, 
                            generator, 
                            latent_dim, 
                            seed,
                            fps,
                            batch_size,
                            strength,
                            max_duration,
                            use_peak)
                    
                    with gr.Row():
                        with gr.Box():
                            with gr.Column():
                                with gr.Row():
                                    wave_in = gr.Audio(
                                        value='examples/example.mp3', 
                                        type="filepath", 
                                        show_label="Music"
                                    )
                                with gr.Row():
                                    seed_in = gr.Number(
                                        value=128, 
                                        label='Seed'
                                    )
                                with gr.Row():
                                    fps_in = gr.Slider(
                                        value=default_fps, 
                                        minimum=5,
                                        maximum=max_fps, 
                                        show_label="FPS"
                                    )
                                with gr.Row():
                                    strength_in = gr.Slider(
                                        value=strength, 
                                        maximum=1, 
                                        show_label="Strength"
                                    )
                                with gr.Row():
                                    max_duration_in = gr.Slider(
                                        value=default_max_duration, 
                                        minimum=5,
                                        maximum=max_duration, 
                                        show_label="Max Duration"
                                    )
                                
                                with gr.Row():
                                    peak_in = gr.Checkbox(value=use_peak, show_label="Use peak")

                                with gr.Row():
                                    generate_button = gr.Button('Generate')
                        with gr.Box():
                            with gr.Column():
                                with gr.Row():
                                    interpolated_video = gr.Video(label='Output Video')

                    
                    generate_button.click(interpolate,
                                        inputs=[
                                            wave_in,
                                            seed_in,
                                            fps_in,
                                            strength_in,
                                            max_duration_in,
                                            peak_in
                                        ],
                                        outputs=interpolated_video)
        
        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.beat-interpolator" alt="visitor badge"/></center>'
        )

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()

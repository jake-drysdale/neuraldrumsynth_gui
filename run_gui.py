#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:09:34 2023

@author: jake
"""


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import soundfile as sf

# local imports
from inference import load_models, layer_noise, feature_slider, args
import pylib as py






if __name__ == "__main__":
    


    # =============================================================================
    # create generations folder
    # =============================================================================
    def make_date_filename():
        # datetime object containing current date and time
        now = datetime.now()
        
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        return dt_string.replace("/","-").replace(":","-").replace(" ","_")
    
    
    
    #get directories
    generator_ckpt_dir = '/media/jake/ubuntu_storage/thesis_final_evaluation/style-drumsynth/output/style_cond_mapping_z64_scale2_thesis_16k_8912_z100_tomatchstylecond_this/checkpoints' 
    kick_directions = py.join(os.getcwd(), 'fixed_settings/kick6.npy')
    snare_directions = py.join(os.getcwd(), 'fixed_settings/snare6.npy')
    cymbal_directions = py.join(os.getcwd(), 'fixed_settings/cymbal6.npy')
    generations_dir= py.join(os.getcwd(), 'generations' )
    kick_umap = py.join(os.getcwd(), 'fixed_settings/kick_umap.npy')
    snr_umap = py.join(os.getcwd(), 'fixed_settings/snare_umap.npy')
    cymb_umap = py.join(os.getcwd(), 'fixed_settings/cymbal_umap.npy')
    
    generations_flns = make_date_filename()
    
    os.mkdir(os.path.join(generations_dir, generations_flns))
    
    save_dir = os.path.join(generations_dir, generations_flns)
    
    # load networks
    G, S = load_models()
    
    
    drum_types=['kick', 'snare', 'cymbal']
    
    
    
    
    
    
    
    @tf.function
    def sample(styles, ones, inc_noise, G):
        return G(styles + [ones, inc_noise], training=False)
    
    def w_average(styles, snare_label, damp_label, pos_label):
        return K.mean(S([styles, snare_label, damp_label, pos_label]), axis=[0], keepdims=True)
    
    
    def truncation_trick(styles, labels, w_average, psi):
        w_spaces=[]
        for i in range(len(styles)):
            w = S([styles[i], labels[0], labels[1], labels[2]])
            w_new =  w_average + psi * (w-w_average)
            w_spaces.append(w_new)
        
        return w_spaces
    
    
    noise_input = layer_noise(1)
    def generate():
        global noise_input
        noise_input = layer_noise(1)
    
    
    n_save = noise_input
    w_test = S([n_save[0], np.array([1])])
    
    umap_w = w_test
    
    
    
    # mixed_latent_vector = None
    
    
    umap_ratio=1
    
    
    # Callback function for slider change
    def on_slider_change(value):
        global umap_ratio
        umap_ratio = float(value)
        get_umap_emedding()
        generate_drum()
        do_plot(waveform=waveform_and_w_spaces)
    
    
    
        
        
    root = tk.Tk()
    root.title('Neural Drum Synth')
    root.geometry('750x610+10+10')
    
    gen_num=tk.IntVar()
        
        
    
    layer_activations=[1,1,1,1,1,1,1,1]
    
    low_button_value = tk.BooleanVar()
    mid_button_value = tk.BooleanVar()
    high_button_value = tk.BooleanVar()
    
    button_values=[low_button_value, mid_button_value, high_button_value]
    button_values[0].set(True)
    button_values[1].set(True)
    button_values[2].set(True)
    
    
    def get_layer_activation():
        global layer_activations
        # button_status is a tuple of three boolean values: (low, mid, high)
    
        # Initialize an empty list to store the layer activations
        layer_activations = []
    
        # Check the status of each button and set the layer activations accordingly
        for i in range(8):
            if low_button_value.get() and i < 3:  # Low button active and in layers 1-2
                layer_activations.append(1)
            elif mid_button_value.get() and 3 <= i < 5:  # Mid button active and in layers 3-5
                layer_activations.append(1)
            elif high_button_value.get() and i >= 5:  # High button active and in layers 6-8
                layer_activations.append(1)
            else:
                layer_activations.append(0)  # Inactive layers
        # print(layer_activations)
        # return layer_activations
    
    
    
    
    
    
        
    low_fixed_latent = None
    mid_fixed_latent = None
    high_fixed_latent = None
    
    
    
    def synthesize(drum_label, G, S,
                   prinicpal_directions, direction_slider, noise_input, inc_mix):
        drum_gen = np.array([drum_label])
        global umap_w
        global umap_ratio
        global low_fixed_latent
        global mid_fixed_latent
        global high_fixed_latent
        global layer_activations
        global button_values
        
        # layer_activations = get_layer_activation(band_buttons)
        
        prinicpal_directions = prinicpal_directions[drum_label]
        const_save = np.ones((1, 1), dtype=np.float32)
        inc_n_save = np.random.uniform(0.0, inc_mix,
                                       size = [1, 
                                               args.audio_length, args.channels])
        
        
        # n_save = layer_noise(1)
        n_save = noise_input
        w_test = S([n_save[0], drum_gen])
        w_test = w_test * (1 - umap_ratio) +  umap_w * umap_ratio
    
        
        w_hats=[]
        for i in range(len(prinicpal_directions)):
            w_hat = feature_slider(w_test, prinicpal_directions, i, direction_slider[i])
            w_hats.append(w_hat)
            
        concat_w = tf.concat(w_hats,0)
        concat_w_hat = tf.expand_dims(tf.reduce_sum(concat_w,0),0)
        
        if button_values[0].get():
            low_fixed_latent = concat_w_hat
    
        if button_values[1].get():
            mid_fixed_latent = concat_w_hat
    
        if button_values[2].get():
            high_fixed_latent = concat_w_hat
                
        
        w_mod = []
        for i in range(args.num_res - 1):
            if layer_activations[i]:  # If the layer is activated
                w_mod.append(concat_w_hat)
            else:  # If the layer is deactivated, use the fixed latent value
                if i < 2 and low_fixed_latent is not None:
                    w_mod.append(low_fixed_latent)
                elif 2 <= i < 5 and mid_fixed_latent is not None:
                    w_mod.append(mid_fixed_latent)
                elif i >= 5 and high_fixed_latent is not None:
                    w_mod.append(high_fixed_latent)
                else:
                    w_mod.append(concat_w_hat)
    
        # w_mod=[]
        # for i in range(args.num_res-1):
        #     w_mod.append(concat_w_hat)
        
            
        return sample(w_mod, const_save, inc_n_save, G)[0,:,:], concat_w_hat
    
    
    
    component = 0
    direction_slider = 0
    cons_noise_amount=1
    prinicpal_directions = [np.load(kick_directions), np.load(snare_directions), np.load(cymbal_directions)]
    umap_spaces = [np.load(kick_umap), np.load(snr_umap), np.load(cymb_umap)]
    
        
        
        
        
        
        
        
    # =============================================================================
    #     GUI STUFF
    # =============================================================================
    
    
    
    
    
    
    
    
    
        
    
        
    slider_var1 = tk.IntVar()
    slider_var2 = tk.IntVar()
    slider_var3 = tk.IntVar()
    slider_var4 = tk.IntVar()
    slider_var5 = tk.IntVar()
    slider_var6 = tk.IntVar()
    
    
    slider_vars=[slider_var1, slider_var2, slider_var3, slider_var4, slider_var5, slider_var6] #for pca sliders
    
    drum_var=tk.IntVar() #for changing condition
    
    
    master_level=tk.DoubleVar()
    master_level.set(1.0)
    
    
    inc_level = tk.DoubleVar()
    inc_level.set(1.0)
    
    
    
    # =============================================================================
    # pca sliders
    # =============================================================================
    
    
    
    pca_scale = 80
    
    frame_pca = ttk.Frame(root)
    frame_pca.place(x=500, y=240, width=250, height=110)  # Increased height to accommodate the label
    
    w1_label = tk.Label(frame_pca, text='Morphing Parameters', font=('calibre', 10, 'bold'))
    w1_label.pack(side="top")
    
    slider_vars = []
    pca_sliders = []
    for i in range(6):
        slider_vars.append(tk.IntVar())
        w1 = tk.Scale(
            frame_pca,
            from_=-pca_scale, to=pca_scale,
            width=10, sliderlength=20,
            variable=slider_vars[i],
            resolution=2, showvalue=0,
            tickinterval=10,
            command=lambda x: [generate_drum(), do_plot(waveform=waveform_and_w_spaces)]
        )
        w1.pack(side="left", pady=5)
    
     # tickinterval=4
    
    # w1_label = tk.Label(root, text = 'Morphing Parameters', font=('calibre',10, 'bold'))
    # w1_label.place(x=520, y=250, width=200, height=20)
    
    
    
    # pca_scale=30
    
    # # slider_vars=[]
    # pca_sliders=[]
    # for i in range(6):
    #     # slider_vars.append(tk.IntVar())
    #     w1 = tk.Scale(root, from_=-20, to=20, width=10, sliderlength=20, tickinterval=4, variable=slider_vars[i],
    #                   resolution=1, showvalue=0, command= lambda x: [generate_drum(), do_plot(waveform=waveform_and_w_spaces)]
    #     )
    #     w1.place(x=500+(i*40), y=260, width=40, height=90)
    
    
    
    
    
    def synth(drum_var_label,slider, inc_mix_level):
    
        x = synthesize(drum_var_label ,G, S, prinicpal_directions, slider, noise_input, inc_mix_level)
    
            
        x = x[0]*master_level.get()
        return x
    
    
    
    
    
    
    waveform_and_w_spaces = synth(drum_var.get(), [slider_vars[0].get(), 
                                                  slider_vars[1].get(),
                                                  slider_vars[2].get(),
                                                  slider_vars[3].get(),
                                                  slider_vars[4].get(),
                                                  slider_vars[5].get()],inc_level.get())
    
    
    def generate_drum():
        global waveform_and_w_spaces
        waveform_and_w_spaces = synth(drum_var.get(), [slider_vars[0].get(), 
                                                  slider_vars[1].get(),
                                                  slider_vars[2].get(),
                                                  slider_vars[3].get(),
                                                  slider_vars[4].get(),
                                                  slider_vars[5].get()],inc_level.get())
    
    # def synthesize_in_thread(drum_label, slider_values, inc_mix_level):
    #     waveform = synth(drum_label, slider_values, inc_mix_level)  # Your existing synthesis function
    #     do_plot(waveform=waveform)
    
    
    
    # =============================================================================
    # audio rate and silence
    # =============================================================================
    
    
    silence_zeros = tf.expand_dims(np.zeros(20000, dtype='float32'), 1)
    end_zeros = tf.expand_dims(np.zeros(10000, dtype='float32'), 1)
    
    new_sample_rate = tk.IntVar()
    new_sample_rate.set(16000)
    
    
    def get_updated_samples(sr):
        return int(8192 * (sr / 16000))
        
    
    # =============================================================================
    # plot
    # =============================================================================
    
    def do_plot(waveform=waveform_and_w_spaces,
                silence=silence_zeros):
    
        # x = synthesize(snare_label, damp_label, pos_label, G, S, prinicpal_directions, slider, noise_input)
        ax.clear()
        ax.set_facecolor("black")
        new_wave = tf.concat([waveform_and_w_spaces, end_zeros], 0)
        ax.plot(new_wave, 'springgreen')
        ax.set_ylim(-1,1)
        # ax.set_xlim(0,zoom_slider.get())
        ax.set_xlim(0,get_updated_samples(new_sample_rate.get()))
        ax.set_xlabel('samples', fontsize=10)
        ax.set_ylabel('amplitude', fontsize=10)
        ax.set_title('Waveform', fontsize=15)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        # plt.axis('off')
        # plt.grid(b=None) 
        canvas.draw()
        
    #     # new_wave = tf.concat([silence, waveform], 0)
    
    #     # sd.play(new_wave[:,0], 16000)
    
    
    
    
    
    
    
    frame1 = tk.Frame(root); frame1.place(x=0, y=0, width=500, height=500)
    figure = plt.Figure(figsize=(5,5), facecolor='silver')
    canvas = FigureCanvasTkAgg(figure, frame1)
    canvas.get_tk_widget().place(x=0,y=0,width=500,height=500)
    ax = figure.add_subplot(111)
    
    
    # =============================================================================
    # soma logo
    # =============================================================================
    import matplotlib.image as mpimg
    soma_logo = mpimg.imread('./logo-bw.png')
    
    figure3 = plt.Figure(figsize=(4,4), facecolor='gainsboro')
    canvas3 = FigureCanvasTkAgg(figure3, root)
    canvas3.get_tk_widget().place(x=630,y=520,width=120,height=88)
    ax3 = figure3.add_subplot(111)
    ax3.set_position([0, 0, 1, 1])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.axis('off')
    ax3.xaxis.label.set_color('lightgrey')
    ax3.yaxis.label.set_color('lightgrey')
    
    ax3.set_facecolor("gainsboro")
    ax3.imshow(soma_logo)
    canvas3.draw()
    
    
    
    
    
    
    # =============================================================================
    # zoom /play/ rand buttons
    # =============================================================================
    
    
    frame3 = tk.Frame(root)
    frame3.place(x=0, y=510, width=500, height=130)
    
    # # Create a label for "Zoom" text
    # zoom_label = tk.Label(frame3, text="Zoom", font=("calibri", 10, "bold"))
    # zoom_label.place(x=100, y=47, width=90, height=20)
    
    # # Create a zoom slider widget
    # zoom_slider = tk.Scale(frame3, from_=8192, to=100, resolution=1, orient='horizontal', showvalue=0)
    # zoom_slider.place(x=100, y=65, width=90, height=20)
    # zoom_slider.set(8192)  # Set the default value to max zoom (8192 samples)
    
    # def on_zoom_change(value):
    #     global waveform_and_w_spaces
    #     zoom_level = int(value)
    #     ax.set_xlim(0, zoom_level)  # Update the x-axis limits based on the zoom level
    #     do_plot(waveform=waveform_and_w_spaces)  # Update the plot
    
    #     # Update the slider value label
    #     slider_value_label.config(text=str(zoom_level))
    
    # # Bind the zoom slider to the callback function
    # zoom_slider.config(command=on_zoom_change)
    
    # # Create a label to display the slider value below the slider
    # slider_value_label = tk.Label(frame3, text="8192", font=("calibri", 10))
    # slider_value_label.place(x=100, y=85, width=90, height=20)
    
    
    
    
        
        
    
    
    frame2 = tk.Frame(root); frame2.place(x=520, y=0, width=1000, height=400) #umap frame
    
    
    
    
    
    do_plot()
    
    gen_num=tk.IntVar()
    gen_num.set(0)
    
    def save_snare(x):
        global new_sample_rate
        filename = (drum_types[drum_var.get()] + '_' + str(gen_num.get()))
        
        sf.write(os.path.join(save_dir,filename+'.wav'), x, new_sample_rate.get())
        gen_num.set(gen_num.get()+1)
    
    
    
    
    
    def play_audio():
        global waveform_and_w_spaces
        global new_sample_rate
        new_wave = tf.concat([silence_zeros, waveform_and_w_spaces], 0)
        sd.play(new_wave[:,0].numpy(), new_sample_rate.get())
    
    # =============================================================================
    # play back button
    # =============================================================================
    
    
    nudge_buttons=170
    
    btplot1 = tk.Button(frame3, text='Play', command= lambda: [play_audio()])
    btplot1.place(x=nudge_buttons, y=0, width=80, height=30)
    
    # =============================================================================
    # randomize button
    # =============================================================================
    btplot2 = tk.Button(frame3, text='Rand Seed', command= lambda: [generate(), generate_drum(), do_plot(waveform=waveform_and_w_spaces)])
    btplot2.place(x=nudge_buttons, y=32, width=80, height=30)
    
    
    
    # Create slider and labels
    blend_slider = tk.Scale(frame3, from_=1, to=0, resolution=0.01, showvalue=0, orient='horizontal')
    blend_slider.set(1)
    slider_label = tk.Label(frame3, text='Blend Seed', font=('calibre', 10, 'bold'))
    
    # Place GUI elements
    blend_slider.place(x=95+nudge_buttons, y=40, width=90, height=40)
    slider_label.place(x=95+nudge_buttons, y=21, width=90, height=20)
    
    
    
    # Bind slider to callback
    blend_slider.config(command=on_slider_change)
    
    
    
    # # =============================================================================
    # # save button
    # # =============================================================================
    btplot3 = tk.Button(frame3, text='Save', command= lambda: save_snare(waveform_and_w_spaces))
    
    
    btplot3.place(x=nudge_buttons, y=64, width=80, height=30)
    
    
    
    
    frame_mix = ttk.Frame(root)
    frame_mix.place(x=520, y=420, width=300, height=80)
    
    frame_mix_label = tk.Label(root, text='Amplitude', font=('calibre', 10, 'bold'))
    frame_mix_label.place(x=520, y=405)
    frame_pitch_label = tk.Label(root, text='Pitch', font=('calibre', 10, 'bold'))
    frame_pitch_label.place(x=610, y=405)
    
    inc_label = tk.Label(root, text = 'Noise', font=('calibre',10, 'bold'))
    inc_label.place(x=680, y=405)
    
    
    
    
    
    
    
    master_mix = tk.Scale(frame_mix, from_=1.0, to=0.0, width=8, sliderlength=20, variable=master_level,
                  resolution=0.1, showvalue=0, orient='vertical', command= lambda x: [generate_drum(), do_plot(waveform=waveform_and_w_spaces)])
    
    master_mix.pack(side='left', pady=5, padx=30)
    
    
    
    inc_mixer = tk.Scale(frame_mix, from_=1.0, to=0.0, width=8, sliderlength=20, variable=inc_level,
                  resolution=0.1, showvalue=0, orient='vertical', command= lambda x: [generate_drum(), do_plot(waveform=waveform_and_w_spaces)]
    )
    
    
    
    
    
    
    
    def on_pitch_change(value):
        global waveform_and_w_spaces
        # samples_plot = get_updated_samples(value)
        # ax.set_xlim(0, samples_plot)  # Update the x-axis limits based on the zoom level
        do_plot(waveform=waveform_and_w_spaces)  # Update the plot
    
    
    
    
    
    pitch = tk.Scale(frame_mix, from_=20000, to=10000, width=8, sliderlength=20, variable=new_sample_rate, orient="vertical", showvalue=False, command=do_plot())
    pitch.pack(side='left', pady=5, padx=30)
    pitch.set(16000)
    
    # Bind the pitch slider to the callback function
    pitch.config(command=on_pitch_change)
    
    
    inc_mixer.pack(side='left', pady=5, padx=30)
    
    # load umap values
    
    umap_embedding = umap_spaces[drum_var.get()]
    
    
    frame4 = tk.Frame(root); frame4.place(x=550, y=80, width=150, height=150)
    
    
    cross_size = 5
    cross_lines = []
    
    def get_umap_emedding():
        global umap_embedding
        umap_embedding = umap_spaces[drum_var.get()]
    
    # umap_embedding = kick_embedding
    
    def draw_cross(x, y):
        global cross_lines
        line1 = xy_canvas.create_line(x - cross_size, y, x + cross_size, y, fill="black")
        line2 = xy_canvas.create_line(x, y - cross_size, x, y + cross_size, fill="black")
        cross_lines = [line1, line2]
    
    def clear_cross():
        global cross_lines
        for line in cross_lines:
            xy_canvas.delete(line)
    
    umap_index=0
    
    def on_canvas_click(event):
        global umap_w
        global umap_index
        
        get_umap_emedding()
        x = event.x
        y = event.y
    
        # Adjust the clicked coordinates to the desired range (0-99)
        adjusted_x = int((x / canvas_width) * 9)
        adjusted_y = int((y / canvas_height) * 9)
    
        clear_cross()
        draw_cross(x, y)
    
        # Calculate the index into the test_pts array
        index = adjusted_y * 10 + adjusted_x
        umap_index = index
    
        if index < len(umap_embedding):
            umap_value = umap_embedding[index]
            umap_w=umap_value
            generate_drum()
            do_plot(waveform=waveform_and_w_spaces)
        else:
            print("Index out of bounds")
            
    
    
    def umap_inverse_transform(mapped_x, mapped_y, model):
        umap_point = np.array([[mapped_x, mapped_y]])
        original_point = model.predict(umap_point)
        return original_point
    
    
    def interpolate_umap(normalized_x, normalized_y, umap_embedding):
        x_min, x_max = umap_embedding[:, 0].min(), umap_embedding[:, 0].max()
        y_min, y_max = umap_embedding[:, 1].min(), umap_embedding[:, 1].max()
    
        mapped_x = normalized_x * (x_max - x_min) + x_min
        mapped_y = normalized_y * (y_max - y_min) + y_min
    
        return mapped_x, mapped_y
    
    
    
    tk.Label(frame2, text ="Drum Space", font=('calibre',10, 'bold')).grid(row=2,
                                                                             column=1, pady=5)
    
    
    
    
    
    canvas_width = 150
    canvas_height = 150
    xy_canvas = tk.Canvas(frame4, width=canvas_width, height=canvas_height, bg="white")
    xy_canvas.pack()
    
    # Draw an initial cross at the center
    center_x, center_y = canvas_width // 2, canvas_height // 2
    draw_cross(center_x, center_y)
    
    xy_canvas.bind("<Button-1>", on_canvas_click)
    
    
    
    
    
    
    
    def change_snare(val, text):
        # global snare_var
        global drum_var
        global waveform_and_w_spaces
        global umap_index
        global umap_w
        drum_var.set(val)
        get_umap_emedding()
        umap_value = umap_embedding[umap_index]
        umap_w=umap_value
        generate_drum()
        do_plot(waveform=waveform_and_w_spaces)
        # get_umap_emedding()
        # do_plot(waveform=waveform_and_w_spaces)
    
        # snare_text.config(text=text)
        # print(snare_var.get())
        # global snare_var
        # snare_var = vals
        
    
    tk.Label(frame2, text = 'Drum Type', font=('calibre',10, 'bold')).grid(row=0,
                                                                             column=1, pady=5)
    
    drum_buttons=[]
    for i in range(len(drum_types)):
        button = tk.Button(frame2, text=drum_types[i], command = lambda i=i: [change_snare(i, drum_types[i])])
        drum_buttons.append(button)
        if i <= 4:
            drum_buttons[i].grid(row=(1),column=i, padx=5, sticky="ew")
        # else :
        #     snare_buttons[i].grid(row=(i+1)-5,column=1, padx=5, sticky="ew")
    
    
    
    
    
    frame7 = ttk.Frame(root)
    frame7.place(x=540, y=370)
        
    band_text=["low", "mid", "high"]
    
    
    layer_label = tk.Label(root, text="Layer Select", font=("calibre", 10, "bold"))
    layer_label.place(x=577, y=350)
    
    
    
    # button_values = [tk.BooleanVar(value=True) for _ in range(3)]
    
    
    
    
    
    
    lowbutton = ttk.Checkbutton(
        frame7,
        text=band_text[0],
        variable=button_values[0],
        command=get_layer_activation,
    )
    lowbutton.grid(row=0, column=0, sticky="w", padx=10)
    
    
    
    midbutton = ttk.Checkbutton(
        frame7,
        text=band_text[1],
        variable=button_values[1],
        command=get_layer_activation
    )
    midbutton.grid(row=0, column=1, sticky="w", padx=10)
    
    
    
    hibutton = ttk.Checkbutton(
        frame7,
        text=band_text[2],
        variable=button_values[2],
        command=get_layer_activation
    )
    hibutton.grid(row=0, column=2, sticky="w", padx=10)
    
    
    frame_pca.lift()
    
    root.mainloop()


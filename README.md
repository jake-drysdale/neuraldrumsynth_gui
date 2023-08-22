# Neural Drum Synth GUI
GUI for controlling a drum sample generator.

Please download the pre-trained model checkpoints from the following [link](https://drive.google.com/file/d/1YePxWx2zeJFqEOxOD9k15juVBPHh84bb/view?usp=drive_link).
Unzip the folder and store it in the "output" folder.

This project utilsies a style-based generative adversarial network trained on collection of drum samples containing kick, snare, and cymbals.

The GUI contains the following features:

* __Drum type selection__: kick, snare, cymbal - _achieved by conditioning the generative model on class labels during training_
* __Drum space__: a 2D pad used for exploring drum sounds, with similar sounds presented closer together - _created by applying UMAP to the latent distribution_
* __Morphing parameters__: a set of synthesiser parameters for fine tuning and making variations - _identified using PCA to find directions in the latent space_
* __Layer select__: enables further fine tuning by selecting which layers of the network will be affected by Drum space and Morphing parameters - _low [first three layers], mid [second three layers], high [final three layers] if the buttons are deselected, the current latent variable will be frozen for those layers until selected again for more focused control_  
* __Amplitude__: adjust the global ampltiude level
* __Pitch__: change the drum pitches - _alters the sample rate for playback, plotting and saving_ 
* __Noise__: introduces subtle variations each time a sound is generated - _when maxed out (1.0), it emulates an analog synthesiser, in a drum will have subtle differences each time it is generated, when minimised (0.0), generations will be precise_
* __Rand seed__ : generates a random drum sound that can be manipulated with the Morphing parameters - _randomly samples the latent space_
* __Blend Seed__: enables blending between the random seed and the UMAP space - _interpolates between the random latent vector and the UMAP vector_
* __Play__: audio playback
* __Save__: saves audio to the generations folder

# DeepTurkish
Turkish Implementation of DeepSpeech2. https://arxiv.org/abs/1512.02595 End2End Automatic Speech Recognition.

For the model, we utilized https://www.comet.ml/site/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai/.

For the decoders, we utilized and extended https://github.com/githubharald/CTCDecoder. (log-probability integrated)

For training the model, you can use https://commonvoice.mozilla.org/ for free, however the audio samples here can greatly degrade the training quality, therefore we advice cleaning the samples from the microphone clicks and the very long silence regions. A notebook that can perform this is included in the data_preparation folder.

https://catalog.ldc.upenn.edu/LDC2006S33 is a paid dataset but it will result in faster training.

Also, a wiki-dump formatter for Turkish which is highly conservative is included in the text_formatter.py

# DeepTurkish
Turkish Implementation of DeepSpeech2.

For the model, we utilizied https://www.comet.ml/site/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai/.

For the decoders, we utilized and extended https://github.com/githubharald/CTCDecoder. (log-probability integrated)

For training the model, you can use https://commonvoice.mozilla.org/ , however the audio samples here can greatly degrade the training quality, therefore we advide you to clean the samples from the microphone clicks and the very long silence regions. A code that can do this is included in data_preparation folder.

Also, a wiki-dump formatter for turkish which is highly conservative is included in the text_formatter.py


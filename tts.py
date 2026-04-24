from transformers import VitsModel, AutoTokenizer
import torch
import scipy

model = VitsModel.from_pretrained("facebook/mms-tts-pag")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pag")

text = "Agko anengneng may silew niniman"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# Convert tensor → numpy, squeeze batch dim
output = output.squeeze().cpu().numpy()

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
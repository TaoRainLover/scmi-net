import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model

path_wav2vec2 = '../pretained_model/wav2vec2-base-960h'

processor = Wav2Vec2Processor.from_pretrained(path_wav2vec2)
model = Wav2Vec2ForCTC.from_pretrained(path_wav2vec2)   # 用于ASR等，32维

path_audio = '../dataset/audio_for_test/Ses01F_impro01/Ses01F_impro01_F000.wav'

audio_input, sample_rate = sf.read(path_audio)  # (31129,)
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values  # torch.Size([1, 31129])

logits = model(input_values).logits     # torch.Size([1, 97, 32])
predicted_ids = torch.argmax(logits, dim=-1)    # torch.Size([1, 97])

transcription = processor.decode(predicted_ids[0])  # ASR的解码结果

model = Wav2Vec2Model.from_pretrained(path_wav2vec2)    # 用于提取通用特征，768维
wav2vec2 = model(input_values)['last_hidden_state']     # torch.Size([1, 97, 768])，模型出来是一个BaseModelOutput的结构体。
print(wav2vec2)
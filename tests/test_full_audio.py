#!/usr/bin/env python3
"""Test full-audio (non-streaming) ASR inference to verify audio quality."""
import warnings
warnings.filterwarnings('ignore')
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

MODEL_PATH = '/workspace/models/Parakeet_Reatime_En_600M.nemo'
AUDIO_PATH = '/workspace/tests/fixtures/harvard_16k.wav'

print('Loading model...')
model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location='cuda:0')

# Greedy decoding for Blackwell
model.change_decoding_strategy(
    decoding_cfg=OmegaConf.create({'strategy': 'greedy', 'greedy': {'max_symbols': 10}})
)
model.eval()

# Load full audio
audio_data, sr = sf.read(AUDIO_PATH, dtype='float32')
print(f'Audio: {len(audio_data)/sr:.2f}s @ {sr}Hz')

# Full-audio inference (non-streaming)
audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).cuda()
audio_len = torch.tensor([len(audio_data)]).cuda()

with torch.no_grad():
    processed, processed_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
    print(f'Processed shape: {processed.shape}')

    encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)
    print(f'Encoded shape: {encoded.shape}')

    result = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_len)

    if isinstance(result, tuple):
        best_hyp = result[0]
    else:
        best_hyp = result

    if best_hyp and best_hyp[0]:
        hyp = best_hyp[0]
        if hasattr(hyp, 'text'):
            print(f'Transcription: "{hyp.text}"')
        elif hasattr(hyp, 'y_sequence'):
            tokens = hyp.y_sequence.cpu().numpy().tolist()
            text = model.decoding.decode_tokens_to_str(tokens)
            print(f'Transcription: "{text}"')
        else:
            print(f'Hypothesis: {hyp}')
    else:
        print('No transcription returned')

print('Done!')

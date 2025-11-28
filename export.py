import argparse
import os
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from kokoro import KModel
from export_utils import (
    load_sample, 
    export_model1, 
    export_model2, 
    export_model3, 
    export_model4,
    generate_example_inputs_for_model2_3_and_4
)


class Model1(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.bert = kmodel.bert
        self.bert_encoder = kmodel.bert_encoder
        self.predictor_text_encoder = kmodel.predictor.text_encoder
        self.predictor_lstm = kmodel.predictor.lstm
        self.predictor_duration_proj = kmodel.predictor.duration_proj
    
    def forward(self, input_ids, ref_s, input_lengths, text_mask):
        attention_mask = (~text_mask).to(torch.int32)
        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor_text_encoder(d_en, s, input_lengths, text_mask)
        self.predictor_lstm.flatten_parameters()
        x, _ = self.predictor_lstm(d)
        duration = self.predictor_duration_proj(x)
        return duration, d


class Model2(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.predictor_F0Ntrain = kmodel.predictor
        self.text_encoder = kmodel.text_encoder
    
    def forward(self, en, ref_s, input_ids, input_lengths, text_mask, pred_aln_trg):
        s = ref_s[:, 128:]
        F0_pred, N_pred = self.predictor_F0Ntrain.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        return F0_pred, N_pred, asr


class Model3(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder
        self.n_fft = kmodel.decoder.generator.post_n_fft
        self.hop_length = kmodel.decoder.generator.stft.hop_length
    
    def forward(self, asr, F0_pred, N_pred, ref_s, har):
        x = self.decoder.forward_with_har_raw(asr, F0_pred, N_pred, ref_s[:, :128], har)
        return x


class Model4(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.generator = kmodel.decoder.generator
        self.f0_upsamp = kmodel.decoder.generator.f0_upsamp
        self.m_source = kmodel.decoder.generator.m_source
        self.stft = kmodel.decoder.generator.stft
    
    def forward(self, F0_pred):
        f0 = self.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        return har


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="checkpoints/config.json")
    parser.add_argument("--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx")
    parser.add_argument("--use_real_sample", action="store_true")
    parser.add_argument("--lang_code", "-l", type=str, default='a')
    parser.add_argument("--input_length",type=int,default=96)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--voice", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    kmodel = KModel(config=args.config_file, model=args.checkpoint_path, disable_complex=True)
    kmodel.eval()
    
    input_ids = None
    ref_s = None
    
    if args.use_real_sample:
        input_ids, ref_s, speed = load_sample(kmodel, args.lang_code, args.text, args.voice)
    else:
        input_ids = torch.randint(1, 10, (args.input_length-2,)).numpy()
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
        ref_s = torch.randn(1, 256)
    
    model1 = Model1(kmodel).eval()
    export_model1(model1, args.output_dir, input_ids=input_ids, ref_s=ref_s)
    
    model2_inputs, model3_inputs, model4_inputs = generate_example_inputs_for_model2_3_and_4(kmodel, input_ids, ref_s)
    
    model2 = Model2(kmodel).eval()
    export_model2(model2, args.output_dir, model2_inputs)
    
    model3 = Model3(kmodel).eval()
    export_model3(model3, args.output_dir, model3_inputs)
    
    model4 = Model4(kmodel).eval()
    export_model4(model4, args.output_dir, model4_inputs)
    
    print(f"导出完成: {args.output_dir}")

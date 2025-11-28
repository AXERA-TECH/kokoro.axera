import os
import zipfile as _zip
import io as _io
import numpy as np
import torch
import onnx
import onnxsim


def load_input_ids(pipeline, text):
    if pipeline.lang_code in 'ab':
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = list(filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps)))
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids


def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to('cpu')
    return pack[len(phonemes) - 1]


def load_sample(kmodel, lang_code='a', text=None, voice=None):
    from kokoro import KPipeline
    
    pipeline = KPipeline(lang_code=lang_code, model=kmodel, device='cpu')
    
    if text is None:
        if lang_code == 'a':
            text = 'The sky above the port was the color of television, tuned to a dead channel.'
            voice = voice or 'checkpoints/voices/af_heart.pt'
        elif lang_code == 'z':
            # text = '人工智能正在改变我们的生活方式，让科技变得更加智能和便捷。'
            text = '致力于打造世界领先的人工智能感知与边缘计算芯片。'
            voice = voice or 'checkpoints/voices/zf_xiaoxiao.pt'
        else:
            text = 'Hello world!'
            voice = voice or 'checkpoints/voices/af_heart.pt'
    
    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])
    return input_ids, style, speed


def save_and_zip(name, array, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    if hasattr(array, 'cpu') and hasattr(array, 'numpy'):
        try:
            arr = array.cpu().numpy()
        except Exception:
            arr = np.array(array)
    else:
        arr = np.array(array)
    
    npy_bytes = _io.BytesIO()
    np.save(npy_bytes, arr)
    npy_bytes.seek(0)
    
    zip_path = os.path.join(out_dir, f"{name}.zip")
    with _zip.ZipFile(zip_path, 'w', compression=_zip.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}.npy", npy_bytes.read())


def export_onnx(model, args, input_names, output_names, onnx_file):
    torch.onnx.export(
        model, args=args, f=onnx_file,
        export_params=True, verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    
    try:
        onnx_model_simplified, check = onnxsim.simplify(onnx_model)
        if check:
            onnx_file_sim = onnx_file.replace('.onnx', '_sim.onnx')
            onnx.save(onnx_model_simplified, onnx_file_sim)
    except Exception:
        pass


def export_model1(model, output_dir, input_ids=None, ref_s=None):
    onnx_file = os.path.join(output_dir, "model1_bert_duration.onnx")
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1))
    
    save_and_zip('model1_input_ids', input_ids, output_dir)
    save_and_zip('model1_ref_s', ref_s, output_dir)
    save_and_zip('model1_input_lengths', input_lengths, output_dir)
    save_and_zip('model1_text_mask', text_mask, output_dir)
    
    #fixed_seq_len = input_ids.shape[1]
    # np.save(os.path.join(output_dir, "model1_fixed_seq_len.npy"), np.array(fixed_seq_len))
    
    export_onnx(
        model,
        args=(input_ids, ref_s, input_lengths, text_mask),
        input_names=['input_ids', 'ref_s', 'input_lengths', 'text_mask'],
        output_names=['duration', 'd'],
        onnx_file=onnx_file
    )


def export_model2(model, output_dir, example_inputs):
    onnx_file = os.path.join(output_dir, "model2_f0_n_asr.onnx")
    en, ref_s, input_ids, input_lengths, text_mask, pred_aln_trg = example_inputs
    
    save_and_zip('model2_en', en, output_dir)
    save_and_zip('model2_ref_s', ref_s, output_dir)
    save_and_zip('model2_input_ids', input_ids, output_dir)
    save_and_zip('model2_input_lengths', input_lengths, output_dir)
    save_and_zip('model2_text_mask', text_mask, output_dir)
    save_and_zip('model2_pred_aln_trg', pred_aln_trg, output_dir)
    
    export_onnx(
        model,
        args=example_inputs,
        input_names=['en', 'ref_s', 'input_ids', 'input_lengths', 'text_mask', 'pred_aln_trg'],
        output_names=['F0_pred', 'N_pred', 'asr'],
        onnx_file=onnx_file
    )


def export_model3(model, output_dir, example_inputs):
    onnx_file = os.path.join(output_dir, "model3_decoder.onnx")
    asr, F0_pred, N_pred, ref_s, har = example_inputs
    
    save_and_zip('model3_asr', asr, output_dir)
    save_and_zip('model3_F0_pred', F0_pred, output_dir)
    save_and_zip('model3_N_pred', N_pred, output_dir)
    save_and_zip('model3_ref_s', ref_s, output_dir)
    save_and_zip('model3_har', har, output_dir)
    
    #n_fft = model.n_fft
    #hop_length = model.hop_length
    #np.save(os.path.join(output_dir, "model3_n_fft.npy"), np.array(n_fft))
    #np.save(os.path.join(output_dir, "model3_hop_length.npy"), np.array(hop_length))
    
    export_onnx(
        model,
        args=example_inputs,
        input_names=['asr', 'F0_pred', 'N_pred', 'ref_s', 'har'],
        output_names=['x'],
        onnx_file=onnx_file
    )


def export_model4(model, output_dir, example_inputs):
    onnx_file = os.path.join(output_dir, "model4_har.onnx")
    F0_pred = example_inputs
    
    save_and_zip('model4_F0_pred', F0_pred, output_dir)
    
    export_onnx(
        model,
        args=(F0_pred,),
        input_names=['F0_pred'],
        output_names=['har'],
        onnx_file=onnx_file
    )


def generate_example_inputs_for_model2_3_and_4(kmodel, input_ids=None, ref_s=None):
    if input_ids is None:
        input_ids = torch.randint(1, 100, (76,)).numpy()
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
    if ref_s is None:
        ref_s = torch.randn(1, 256)
    
    speed = 1.0
    
    with torch.no_grad():
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1))
        
        bert_dur = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = kmodel.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = kmodel.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = kmodel.predictor.lstm(d)
        duration = kmodel.predictor.duration_proj(x)
        
        duration_processed = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur_original = torch.round(duration_processed).clamp(min=1).long().squeeze()
        
        original_total_frames = pred_dur_original.sum().item()
        fixed_total_frames = input_ids.shape[1] * 2
        
        scale_factor = fixed_total_frames / original_total_frames
        pred_dur_scaled = torch.round(pred_dur_original.float() * scale_factor).clamp(min=1).long()
        
        diff = fixed_total_frames - pred_dur_scaled.sum().item()
        if diff > 0:
            indices_to_increase = torch.argsort(pred_dur_original, descending=True)[:abs(diff)]
            pred_dur_scaled[indices_to_increase] += 1
        elif diff < 0:
            indices_to_decrease = torch.argsort(pred_dur_scaled, descending=True)
            decreased = 0
            for idx in indices_to_decrease:
                if pred_dur_scaled[idx] > 1 and decreased < abs(diff):
                    pred_dur_scaled[idx] -= 1
                    decreased += 1
                if decreased >= abs(diff):
                    break
        
        pred_dur = pred_dur_scaled
        
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        
        en = d.transpose(-1, -2) @ pred_aln_trg
        model2_inputs = (en, ref_s, input_ids, input_lengths, text_mask, pred_aln_trg)
        
        F0_pred, N_pred = kmodel.predictor.F0Ntrain(en, s)
        t_en = kmodel.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        
        decoder = kmodel.decoder
        gen = decoder.generator
        
        F0_conv_out = decoder.F0_conv(F0_pred.unsqueeze(1))
        N_conv_out = decoder.N_conv(N_pred.unsqueeze(1))
        x_dec = torch.cat([asr, F0_conv_out, N_conv_out], axis=1)
        x_dec = decoder.encode(x_dec, ref_s[:, :128])
        asr_res = decoder.asr_res(asr)
        
        res = True
        for block in decoder.decode:
            if res:
                x_dec = torch.cat([x_dec, asr_res, F0_conv_out, N_conv_out], axis=1)
            x_dec = block(x_dec, ref_s[:, :128])
            if block.upsample_type != "none":
                res = False
        
        f0 = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        har_source, noi_source, uv = gen.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        
        model3_inputs = (asr, F0_pred, N_pred, ref_s, har)
        model4_inputs = F0_pred
    
    return model2_inputs, model3_inputs, model4_inputs


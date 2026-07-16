# DreamZero Text Encoder Official-Weight Compare

This validates PHYAI DreamZero text processor and text encoder against the
official DreamZero implementation with official weights.

## Paths

- Official checkpoint: `/data/share/DreamZero-DROID`
- Local tokenizer: `/data/share/google-umt5-xxl`
- Official phoenix repo: `/data/luyiwen/dreamzero`
- PHYAI repo/container: `/phyai_workspace/phyai` in `phyai_dev_luyiwen`
- Official phoenix dump: `/tmp/dreamzero_text_official_bf16_notf32_phoenix.pt`

## Settings

The comparison uses official text encoder weights under
`action_head.text_encoder.`:

```text
242 tensors loaded
```

Both dump and compare scripts disable TF32 for validation:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

The main run uses `bfloat16`, matching the DreamZero text encoder checkpoint
path/config behavior.

## Latest Phoenix Same-Machine Result

Official dump generated on phoenix with `/data/luyiwen/dreamzero`, compared in
`phyai_dev_luyiwen`:

```text
processor_input_ids:                  max_abs=0
processor_attention_mask:             max_abs=0
processor_negative_input_ids:         max_abs=0
processor_negative_attention_mask:    max_abs=0

action_head.text_encoder. weight_check: checked=242 max_abs=0

text_features:
  shape=(2, 512, 4096) same_shape=True
  max_abs=0.017578125
  mean_abs=2.1316166e-05
  rms_abs=0.00020855594

text_features_active:
  elements=94208
  max_abs=0.017578125
  mean_abs=0.00094903284
  rms_abs=0.0013915815

negative_text_features:
  shape=(2, 512, 4096) same_shape=True
  max_abs=0
  mean_abs=0
  rms_abs=0
```

Interpretation:

- The PHYAI processor/tokenization path matches the official tokenizer exactly
  for the tested prompts and negative prompt.
- The text encoder official-weight load maps all tensors exactly.
- BF16 positive-prompt output is close but not bitwise identical. The remaining
  difference is in BF16 compute/kernel behavior over the full 24-layer T5
  encoder. The empty negative prompt matched exactly in this run.

## Reproduce

Generate the official dump on phoenix host:

```bash
cd /data/luyiwen/dreamzero
CUDA_VISIBLE_DEVICES=0 /data/luyiwen/.conda/envs/dreamzero/bin/python \
  scripts/validation/dump_text_official_weights.py \
  --ckpt-dir /data/share/DreamZero-DROID \
  --tokenizer /data/share/google-umt5-xxl \
  --output /tmp/dreamzero_text_official_bf16_notf32_phoenix.pt \
  --device cuda \
  --dtype bfloat16
```

Copy the dump into the PHYAI container and compare:

```bash
docker cp /tmp/dreamzero_text_official_bf16_notf32_phoenix.pt \
  phyai_dev_luyiwen:/tmp/dreamzero_text_official_bf16_notf32_phoenix.pt

docker exec -w /phyai_workspace/phyai phyai_dev_luyiwen bash -lc \
  'PYTHONPATH=/phyai_workspace/phyai/phyai/src:/phyai_workspace/phyai/phyai-utils-tools/src \
   /phyai_workspace/phyai/.venv/bin/python \
   scripts/validation/compare_text_official_weights.py \
   --ckpt-dir /data/share/DreamZero-DROID \
   --tokenizer /data/share/google-umt5-xxl \
   --dump /tmp/dreamzero_text_official_bf16_notf32_phoenix.pt \
   --device cuda \
   --dtype bfloat16 \
   --diagnostics'
```

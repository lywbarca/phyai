# DreamZero Image Encoder / VAE Official-Weight Compare

This validates PHYAI DreamZero image encoder and VAE against the official
DreamZero implementation with the same official checkpoint and identical inputs.

## Paths

- Official checkpoint: `/data/share/DreamZero-DROID`
- Official dump script: `dump_image_vae_official_weights.py`
- PHYAI compare script: `compare_image_vae_official_weights.py`
- Official dump used here: `/tmp/dreamzero_image_vae_official_notf32.pt`

## Important Numeric Setting

Both scripts explicitly disable TF32 and force math SDPA:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

Without disabling TF32, the image encoder comparison is still close on average,
but patch-token outliers can reach about `3.06e-2` across different machines.

## Latest Result

Official weights, same random input, TF32 disabled:

```text
image_features: shape=(1, 257, 1280) same_shape=True
  max_abs=0.00015377998 mean_abs=1.1443308e-06 rms_abs=2.3987461e-06

vae_encoded: shape=(1, 16, 1, 1, 1) same_shape=True
  max_abs=5.9604645e-07 mean_abs=1.8998981e-07 rms_abs=2.5045478e-07

vae_decoded: shape=(1, 3, 1, 8, 8) same_shape=True
  max_abs=1.1920929e-06 mean_abs=3.8002813e-07 rms_abs=4.8582694e-07

action_head.image_encoder. weight_check: checked=393 max_abs=0
action_head.vae. weight_check: checked=194 max_abs=0
```

## Reproduce

On thor official repo/container:

```bash
docker cp /tmp/dump_image_vae_official_weights.py dreamzero_dev:/workspace/dreamzero/scripts/validation/dump_image_vae_official_weights.py
docker exec dreamzero_dev bash -lc 'cd /workspace/dreamzero && python scripts/validation/dump_image_vae_official_weights.py --ckpt-dir /data/share/DreamZero-DROID --output /tmp/dreamzero_image_vae_official_notf32.pt --device cuda'
docker cp dreamzero_dev:/tmp/dreamzero_image_vae_official_notf32.pt /tmp/dreamzero_image_vae_official_notf32.pt
```

On phoenix PHYAI repo/container after copying the dump to
`/tmp/dreamzero_image_vae_official_notf32.pt`:

```bash
docker cp /tmp/dreamzero_image_vae_official_notf32.pt phyai_dev_luyiwen:/tmp/dreamzero_image_vae_official_notf32.pt
docker cp /tmp/compare_image_vae_official_weights.py phyai_dev_luyiwen:/phyai_workspace/phyai/scripts/validation/compare_image_vae_official_weights.py
docker exec phyai_dev_luyiwen bash -lc 'cd /phyai_workspace/phyai && PYTHONPATH=/phyai_workspace/phyai/phyai/src python scripts/validation/compare_image_vae_official_weights.py --ckpt-dir /data/share/DreamZero-DROID --dump /tmp/dreamzero_image_vae_official_notf32.pt --device cuda --diagnostics'
```

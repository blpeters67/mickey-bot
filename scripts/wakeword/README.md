# Training the "hey DJ" wake-word model

How `assets/wakeword/hey_dj.npz` was made (see `assets/wakeword/README.md` for
what's in it and how well it does). Not used by the bot at run time.

Needs (not in requirements.txt): `piper-tts`, `onnx`, `scipy`, `soundfile`,
`pyarrow`, `torch` (CPU), and these under `$KWS_DIR` (default `/tmp/kws`):

- `valign/` — Piper voices from huggingface.co/rhasspy/piper-voices, patched to
  output phoneme alignments (`python -m piper.patch_voice_with_alignment`)
- `data/noise/*.wav` — background noise (MUSAN noise was used)
- `data/acav_part*.bin` — row ranges of openWakeWord's
  `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (float16, 16×96 per row,
  header 128 bytes; huggingface.co/datasets/davidscripka/openwakeword_features)
- `data/validation_set_features.npy` (same dataset), `data/ls_*.parquet`
  (LibriSpeech from huggingface.co/datasets/openslr/librispeech_asr)

```
python make_data.py train pos 4500 11 $KWS_DIR/feats/train_pos.npz
python make_data.py train adv 4000 12 $KWS_DIR/feats/train_adv.npz
python make_data.py eval pos 500 13 $KWS_DIR/feats/eval_pos.npz
python make_data.py eval adv 500 14 $KWS_DIR/feats/eval_adv.npz
python make_data.py libri $KWS_DIR/data/ls_all_test.other_0000.parquet 2 22 $KWS_DIR/feats/train_libri_other.npz
python make_data.py libri $KWS_DIR/data/ls_all_validation.other_0000.parquet 1 23 $KWS_DIR/feats/eval_libri.npz 1200
STEPS=15000 MINE_EVERY=3000 python train.py $KWS_DIR/model
python evaluate.py $KWS_DIR/model/head.npz
```
(`DISCORD_TOKEN=x` in the environment: make_data imports the bot's front end.)
Then copy `head.npz` to `assets/wakeword/hey_dj.npz` with a `threshold` array added.

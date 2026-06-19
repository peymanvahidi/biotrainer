# Embedding compression tests

Covers the embedding h5 write codec (`EmbeddingService.store_embedding`).

Embeddings used to be written with `gzip`. float32 protein language model
embeddings are dense, full-entropy floats that compress very poorly (~7% here),
while `gzip` is CPU-expensive — so the single I/O worker became the bottleneck
and stalled the GPU. The fix switches the default codec to `lzf` (fast, lossless,
shipped with h5py; reads stay transparent).

See `autoeval-improvements/01-embedding-gzip-compression.md`.

Tests:
- `test_store_embedding_defaults_to_lzf` — the default codec is `lzf`, not `gzip`.
- `test_lzf_round_trip_is_lossless` — embeddings reload bit-for-bit via the normal load path.
- `test_lzf_is_faster_than_gzip` — benchmark proving the speedup on ~30 MB of realistic
  per-residue float32 embeddings (typically ~3x faster to write; prints timings/sizes with `-rP`).

```bash
pytest tests/embedding_compression/ -rP
```

"""
Tests for the embedding h5 write codec.

Background: embeddings were originally written with ``gzip``, then ``lzf``. float16/float32 protein
language model embeddings compress very poorly (~5-10%) while a codec is CPU-expensive, so the single
I/O writer became the bottleneck and stalled the GPU. The default is now ``None`` (no compression,
contiguous storage) — the fastest lossless option on local/fast storage. ``lzf`` (fast, lossless) and
``gzip`` remain available via the ``compression`` parameter when smaller files matter (e.g. a
bandwidth-limited network filesystem). All options are lossless and reads are codec-transparent in h5py.
"""

import os
import time
import tempfile
import unittest

import h5py
import numpy as np

from biotrainer.input_files import BiotrainerSequenceRecord
from biotrainer.embedders import EmbeddingService


def _make_embedding(num_residues: int, dim: int, seed: int) -> np.ndarray:
    """Realistic per-residue float32 embedding.

    Protein language model embeddings are dense, full-entropy floats, which is
    exactly the poorly-compressible case the codec change targets. Full-entropy
    random normals are therefore a faithful (and conservative) proxy.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_residues, dim)).astype(np.float32)


class TestEmbeddingCompression(unittest.TestCase):
    def test_store_embedding_defaults_to_none(self):
        """Default codec is now None: protein-LM embeddings are nearly incompressible, so a codec
        only burns CPU in the I/O writer and stalls the GPU. None writes contiguously (no chunking)."""
        seq_record = BiotrainerSequenceRecord(seq_id="seq_0", seq="MAAGVKL")
        embedding = _make_embedding(num_residues=16, dim=32, seed=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "embeddings.h5")
            with h5py.File(path, "w") as handle:
                EmbeddingService.store_embedding(handle, seq_record, embedding, store_by_hash=False)

            with h5py.File(path, "r") as handle:
                self.assertIsNone(handle["seq_0"].compression)
                self.assertIsNone(handle["seq_0"].chunks)  # contiguous storage

    def test_none_round_trip_is_lossless(self):
        """The new default (no codec) must reproduce the embedding bit-for-bit through the load path."""
        seq_record = BiotrainerSequenceRecord(seq_id="seq_0", seq="MAAGVKLPQ")
        embedding = _make_embedding(num_residues=128, dim=256, seed=11)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "embeddings.h5")
            with h5py.File(path, "w") as handle:
                EmbeddingService.store_embedding(handle, seq_record, embedding, store_by_hash=False)

            loaded = EmbeddingService.load_embeddings(path)

        self.assertIn("seq_0", loaded)
        np.testing.assert_array_equal(loaded["seq_0"].numpy(), embedding)

    def test_none_is_faster_than_lzf(self):
        """The new default (None) must write at least as fast as lzf (it skips compression entirely)."""
        num_sequences, num_residues, dim = 24, 320, 1024
        embeddings = [
            (BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq="MAAGV"),
             _make_embedding(num_residues=num_residues, dim=dim, seed=i))
            for i in range(num_sequences)
        ]

        def store_all(file_path, compression):
            start = time.perf_counter()
            with h5py.File(file_path, "w") as handle:
                for seq_record, emb in embeddings:
                    EmbeddingService.store_embedding(handle, seq_record, emb,
                                                     store_by_hash=False, compression=compression)
            return time.perf_counter() - start

        with tempfile.TemporaryDirectory() as tmp:
            none_time = store_all(os.path.join(tmp, "none.h5"), None)
            lzf_time = store_all(os.path.join(tmp, "lzf.h5"), "lzf")

        print(f"\n[embedding-compression] none={none_time:.3f}s lzf={lzf_time:.3f}s "
              f"(none {lzf_time / none_time:.1f}x faster)")
        # Tolerance guards against scheduler/write-back jitter on shared cluster nodes; the real gap is large.
        self.assertLessEqual(none_time, lzf_time * 1.25,
                             f"Expected None to write at least as fast as lzf (within noise), "
                             f"but none={none_time:.3f}s vs lzf={lzf_time:.3f}s")

    def test_lzf_round_trip_is_lossless(self):
        """lzf must reproduce the embedding bit-for-bit and stay readable through
        the normal load path (h5py decompresses transparently)."""
        seq_record = BiotrainerSequenceRecord(seq_id="seq_0", seq="MAAGVKLPQ")
        embedding = _make_embedding(num_residues=128, dim=256, seed=7)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "embeddings.h5")
            with h5py.File(path, "w") as handle:
                EmbeddingService.store_embedding(handle, seq_record, embedding,
                                                 store_by_hash=False, compression="lzf")

            loaded = EmbeddingService.load_embeddings(path)

        self.assertIn("seq_0", loaded)
        np.testing.assert_array_equal(loaded["seq_0"].numpy(), embedding)

    def test_lzf_is_faster_than_gzip(self):
        """The point of the change: writing realistic float32 embeddings is
        faster with lzf than with gzip. Uses ~31 MB of per-residue embeddings so
        the codec cost dominates measurement noise."""
        num_sequences, num_residues, dim = 24, 320, 1024
        embeddings = [
            (BiotrainerSequenceRecord(seq_id=f"seq_{i}", seq="MAAGV"),
             _make_embedding(num_residues=num_residues, dim=dim, seed=i))
            for i in range(num_sequences)
        ]
        total_mb = num_sequences * num_residues * dim * 4 / (1024 ** 2)

        def store_all(file_path, compression):
            start = time.perf_counter()
            with h5py.File(file_path, "w") as handle:
                for seq_record, emb in embeddings:
                    EmbeddingService.store_embedding(handle, seq_record, emb,
                                                     store_by_hash=False, compression=compression)
            return time.perf_counter() - start

        with tempfile.TemporaryDirectory() as tmp:
            gzip_path = os.path.join(tmp, "gzip.h5")
            lzf_path = os.path.join(tmp, "lzf.h5")

            gzip_time = store_all(gzip_path, "gzip")
            lzf_time = store_all(lzf_path, "lzf")

            gzip_size_mb = os.path.getsize(gzip_path) / (1024 ** 2)
            lzf_size_mb = os.path.getsize(lzf_path) / (1024 ** 2)

        speedup = gzip_time / lzf_time if lzf_time > 0 else float("inf")
        print(f"\n[embedding-compression benchmark] payload={total_mb:.1f} MB float32"
              f"\n  gzip: write {gzip_time:.3f}s, file {gzip_size_mb:.1f} MB"
              f"\n  lzf : write {lzf_time:.3f}s, file {lzf_size_mb:.1f} MB"
              f"\n  -> lzf is {speedup:.1f}x faster to write"
              f" (gzip shrank the file by only {(1 - gzip_size_mb / total_mb) * 100:.0f}%)")

        self.assertLess(lzf_time, gzip_time,
                        f"Expected lzf to write faster than gzip, "
                        f"but lzf={lzf_time:.3f}s vs gzip={gzip_time:.3f}s")


if __name__ == "__main__":
    unittest.main()

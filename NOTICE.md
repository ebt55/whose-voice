# Third-party notices

This repository is MIT licensed (see [LICENSE](LICENSE)). The material below is
not covered by that copyright.

This repository builds on work by others. The following are not covered by the
copyright above.

## 1. Phantom Transfer — Draganov, Dur, Bhongade & Phuong (2026)
   https://github.com/tolgadur/phantom-transfer  (MIT License)

   The poisoned corpora this project analyses are theirs, and are NOT
   redistributed here: `scripts/download_data.py` and the reproduction
   instructions fetch them from the upstream release. The derived prompt pools
   are likewise not committed; `configs/matched_pool_manifest.json` records
   only SHA-256 digests so a re-deriver can verify they obtained the same set.

   One piece of their content is reproduced directly: the verbatim teacher
   prompts used to generate each poisoned corpus appear in
   `configs/personas.yaml`, where they define the D0 oracle condition. They are
   transcribed from their repository and attributed there in place. Their MIT
   copyright and permission notice apply to that material.

   The surface-marker regexes in `src/whosevoice/detectors/lexical.py` are
   written for this project and are deliberately NOT their pattern lists, which
   are broad enough to match ordinary English.

2. Model organisms — Alamerton/sl-organism-{a,b,c}-7b (Apache-2.0), released
   for the Secret Loyalties Hackathon and fine-tuned from Qwen/Qwen2.5-7B-
   Instruct. Not redistributed here; weights are fetched from HuggingFace.

3. Encoders used as instruments, each under its own licence and fetched from
   HuggingFace rather than vendored: sentence-transformers/all-MiniLM-L6-v2 and
   all-mpnet-base-v2 (Apache-2.0), BAAI/bge-base-en-v1.5 and bge-large-en-v1.5
   (MIT), intfloat/e5-base-v2 (MIT), Qwen/Qwen2.5-{0.5B,1.5B}-Instruct
   (Apache-2.0).

The user prompts inside the analysed corpora originate from the Stanford Alpaca
instruction set (CC BY-NC 4.0). No prompt text is redistributed by this
repository — see note 1 above on the manifest.

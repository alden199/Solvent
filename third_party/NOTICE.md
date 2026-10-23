# Third-Party Sources

The bundled frameworks retain their original LICENSE files and copyright
notices. Project licensing does not replace these third-party terms.

| Component | Source | Bundled changes |
| --- | --- | --- |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory | SFT loss masking, weighting and entropy support. |
| Slime | https://github.com/THUDM/slime | Scalar reward-component logging; Solvent rollout/reward hooks live in the project package. |
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM | Slime integration changes, recorded in `slime/patches/megatron.patch` and already applied. |
| SGLang patch | https://github.com/sgl-project/sglang | Integration changes in `slime/patches/sglang.patch`; apply to the pinned source as described in the root README. |

The Megatron patch base is `3714d81d418c9f1bca4594fc35f9e8289f652862`;
the SGLang patch base is `24c91001cf99ba642be791e099d358f4dfe955f5`.
Dependency licenses accompany the patches. `slime/patches/source_fingerprints.json`
is required by the RL launcher's runtime compatibility checks.

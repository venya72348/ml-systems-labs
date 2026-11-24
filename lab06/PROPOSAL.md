# Bilingual, Dialect-Robust On-device Keyword Spotting (English + Ukrainian)

**Student:** Veniamin Vyrovyi, ID 72348  **Course:** ML Systems, individual project proposal

**Problem.** Always-on voice interfaces (smart speakers, wearables, embedded assistants) must recognise short spoken commands locally, without sending audio to the cloud, on hardware with little memory and compute. Most such systems support only a few major languages and work noticeably worse for speakers with regional dialects or accents. The goal is a single compact keyword-spotting (KWS) model that recognises commands in **English and Ukrainian**, works **equally well across Ukrainian regional dialects and Ukrainian-accented English**, and still meets tight latency and size budgets. A further question is whether compression (quantization) hurts the lower-resource language or minority dialects more than the majority group.

**Datasets.**
- **English:** Google Speech Commands v2 (~105k one-second 16 kHz clips, ~2.6k speakers, official train/val/test lists). 10 keywords: *yes, no, up, down, left, right, on, off, stop, go*.
- **Ukrainian:** Multilingual Spoken Words Corpus (MSWC, MLCommons, CC-BY 4.0), Ukrainian subset: one-second word clips extracted from Mozilla Common Voice by forced alignment. 10 command words chosen by available clip count from the MSWC metadata; candidates: *так, ні, стоп, вперед, назад, ліворуч, праворуч, увімкни, вимкни, далі*. Split by speaker so that no speaker appears in more than one split.
- **Classes:** 10 English + 10 Ukrainian keywords + "unknown" (other words from both languages) + "silence" = **22 classes**, one shared model.
- **Dialect/accent test set (own recordings, test only).** Groups follow the three main Ukrainian dialect families, plus two practical variants:
  1. *Northern (Polissian):* Zhytomyr, Chernihiv, northern Kyiv region.
  2. *South-Western:* Galicia (Lviv, Ivano-Frankivsk), Volhynia, Podillia, Transcarpathia.
  3. *South-Eastern:* Poltava, Cherkasy, Slobozhanshchyna (Kharkiv), the South (Zaporizhzhia, Odesa).
  4. *Surzhyk:* mixed Ukrainian-Russian everyday speech.
  5. *Ukrainian-accented English* (English commands only).

  Protocol: each speaker says all 20 commands 3 times, recorded on a phone (16 kHz mono) in two conditions, quiet and with background noise. Recorded with consent; only the speaker's region/group is stored, no names. Target: ≥ 3 speakers per group (~15 speakers, ~900 clips). If a group cannot reach 3 speakers, it is merged with the closest group or reported separately as low-confidence. Where Common Voice metadata carries accent/region labels, those clips are used as an additional test set.
- **Features:** 40 log-mel bins over 1 s, 10 ms hop, computed from raw audio with torchaudio.

**Success metrics.**
- Top-1 accuracy: **≥ 90% on English** test, **≥ 85% on Ukrainian** test; also macro-F1 and confusion matrices.
- **Robustness (fairness gap):** accuracy difference between the best and worst dialect/accent group **≤ 5 pp** on the own test set.
- **Compression fairness:** after optimization, accuracy drops by **≤ 1 pp in every language and dialect group**, not only on average.
- A model counts as successful **only if it also meets both deployment constraints** below.

**Deployment constraints.**
- **Latency ≤ 50 ms** per 1-second clip on a laptop CPU, single thread, batch size 1, **including feature extraction** (median over 100+ runs after warm-up).
- **Model file size ≤ 5 MB.**

**Baseline.** A compact DS-CNN (depthwise-separable convolutions) on log-mel spectrograms, trained in FP32 as the reference, plus a larger "teacher" CNN. Comparison: one bilingual model vs. two monolingual models of the same total size.

**Planned optimization.** Primary: **int8 quantization** (post-training static quantization in PyTorch; quantization-aware training if PTQ loses more than 1 pp in any group). Comparison point: **knowledge distillation** from the teacher into the compact student (as in Lab 7). Accuracy, per-group accuracy, size and latency are reported before and after. Stretch goal: few-shot addition of a new keyword from 5–10 examples.

**Risks and mitigation.**

| Risk | Mitigation |
|---|---|
| MSWC labels come from automatic forced alignment and contain mis-cut clips | Filter clips by duration/energy, manually spot-check a sample per word |
| Far less Ukrainian data than English | Per-language class weighting or balanced sampling; augmentation |
| "unknown" class much larger than each keyword | Subsample "unknown" or weight it in the loss |
| Own dialect set is small, so per-group accuracy is noisy | Use it for testing only; report confidence intervals and per-speaker results |
| Weak noise robustness | Background-noise augmentation and random time shift |
| Latency dominated by feature extraction | Measure feature extraction and model latency separately |

**Deliverables.** Reproducible pipeline (Lab 6 YAML, checkpoints, MLflow); FP32 bilingual baseline; optimized model; results table (overall, per-language and per-dialect accuracy, size, latency); short report including the compression-fairness analysis.

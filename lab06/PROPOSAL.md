# On-device Keyword Spotting under Tight Latency and Size Budgets

**Student:** Veniamin Vyrovyi, ID 72348  **Course:** ML Systems, individual project proposal

**Problem.** Always-on voice interfaces (smart speakers, wearables, embedded assistants) must recognise short spoken commands locally, without sending audio to the cloud, on hardware with little memory and compute. The goal is a keyword-spotting (KWS) model that classifies a 1-second audio clip into one of a small set of commands quickly and with a small footprint.

**Dataset.** Google Speech Commands v2: about 105k one-second, 16 kHz utterances of 35 words from about 2.6k speakers. I use the standard 12-class setup: 10 keywords (*yes, no, up, down, left, right, on, off, stop, go*), plus "unknown" (all other words) and "silence" (cut from the background-noise clips). I use the official train/validation/test split lists. Input features are 40 log-mel bins (or MFCCs) over 1 s with a 10 ms hop.

**Success metric.** The primary metric is top-1 accuracy on the official test split, with a target of **≥ 90%**. Secondary metrics are macro-F1 and the confusion matrix. A model counts as successful **only if it also meets both deployment constraints** below.

**Deployment constraints.**
- **Latency ≤ 50 ms** per 1-second clip on a laptop CPU, with a single thread and batch size 1, **including feature extraction**. I report the median over 100+ runs after warm-up.
- **Model file size ≤ 5 MB.**

**Baseline.** The baseline is a compact CNN on log-mel spectrograms in the DS-CNN style (depthwise-separable convolutions), first trained in FP32 as the reference. I also train a larger "teacher" CNN.

**Planned optimization.** The primary method is **quantization**: post-training static int8 quantization in PyTorch, switching to quantization-aware training if PTQ loses more than 1 pp of accuracy. The comparison point is **knowledge distillation** from the larger teacher into the compact student. For every model I report accuracy, size and latency before and after optimization.

**Risks and mitigation.**

| Risk | Mitigation |
|---|---|
| Class imbalance: "unknown" is far larger than each keyword | Subsample the "unknown" class or weight it in the loss |
| Weak noise robustness | Background-noise augmentation and random time shift during training |
| Latency dominated by feature extraction rather than the network | A torchaudio pipeline that computes features from raw audio (no precomputed features), with its latency measured separately from the model |

**Deliverables.**
1. A reproducible training pipeline: the Lab 6 YAML config, checkpointing and MLflow setup.
2. The FP32 baseline.
3. The optimized model.
4. A results table with accuracy, size and latency.
5. A short report.

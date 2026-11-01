# Decoder-aware improvement cycle: preregistered local pilot

30 September 2026. One primary intervention: add 0.02 times teacher-forced
token CE through the frozen AE decoder to the existing velocity-MSE objective.
This makes latent errors costly when they impede recovery of the supplied text;
it is not a direct semantic objective. High-noise ambiguity may make this loss
counterproductive. No claim of improvement is made before evaluation.

Three arms: unchanged frozen EMA source, continued velocity-only control, and
velocity-plus-decoder-CE treatment. Continued arms initialize from identical
50,000-step EMA weights with identical fresh AdamW optimizers (LR 2e-5),
training indices/noise/dropout seeds, and EMA .99. Changing EMA/LR is shared
pilot infrastructure, not attributed to the treatment. Historical optimizer
state is deliberately not resumed across the changed objective. Subsequent
within-arm checkpoints resume optimizer/model/EMA states under a locked protocol.
The AE and latent representation stay unchanged: no diffusion retraining from
scratch is required to test this fine-tuning intervention.

Local budget: external hard limit 1800 seconds including loading and evaluation.
Each continued arm: at most 60 updates or 240 training seconds, microbatch 1,
accumulation 1; 128 randomly selected training passages, 16 validation passages.
Cache selection/training seed 24680. No test evaluation. If wall time prevents
matched update counts, results are not an eligible comparison. No alternative
training intervention will be tested in this budget; negative or inconclusive
findings will not trigger an unbounded search.

Save every 10 updates and at the end, atomically; exercise model/EMA/optimizer
save-load round-trip at the first periodic save. Stop before optimizer updates
on non-finite losses/gradients and preserve the last valid checkpoint. SIGTERM
requests checkpointing at the next step boundary. Check source hashes and
training protocol on explicit resume. Never reuse an existing directory without
--resume. Finite-loss smoke success is distinct from language quality.

Generation: identical seeds 142-149, DDIM 50, FP32, beam 4, maximum length 128,
minimum 5, repetition penalty 1.2, no-repeat trigrams 3. Unedited final text only.
External scoring: ai-forever/mGPT revision
40897bd7c8b47a76802c411108ca6220438b8b40, FP32, context 512, stride 256.
Record length, EOS/cap flags, regex diversity and repetition. Prepare shuffled
blinded CSV and separate key; AI assessment is not fluent-human assessment.

Success before long-run promotion: in a fresh blinded fluent-human review,
treatment mean grammar and coherence each >=0.5 points better on a five-point
scale than continued control, >=60% paired preference, no material diversity
collapse or generic shortening; automatic PPL supports but does not replace
this. Eight local seeds cannot establish general effectiveness. A cluster pilot
must confirm on at least 30 paired outputs before any large continuation.

Local decision: reject practical efficacy on this budget if treatment visibly
degrades meaning, collapses or numerical safety fails. If finite but quality is
inconclusive, deliver a bounded controlled cluster pilot, not a recommended long
training run. Fine-tuning may need more than 60 updates; this possibility is a
remaining risk, not evidence that extra training will fix it.

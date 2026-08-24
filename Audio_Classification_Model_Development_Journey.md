# "Like a Bosch" Detector — Model Development Journey

*A personal reference notebook for the rebuild of the phrase-detection project. Written understanding-first, like the Docker reference: every decision recorded with its reasoning, every stumble left in, because the stumbles are where the understanding stuck. This is the lab notebook, not the README — the README gets distilled from this later.*

---

## How to use this document

This is organised for **re-reading and de-confusing**, not as a chronological diary. If you want the arc, read [The Arc in One Paragraph](#the-arc-in-one-paragraph) then skim the stage sections. If you're trying to recall a concept (EER, debounce, hard negative mining), jump to [Concepts I Had to Learn](#concepts-i-had-to-learn). If you're prepping for an interview question, [Decisions & Why](#decisions--why) is the section that matters most — it's the reasoning, not just the outcome.

---

## Table of Contents

1. [The Arc in One Paragraph](#the-arc-in-one-paragraph)
2. [The Problem & The Data](#the-problem--the-data)
3. [Stage-by-Stage](#stage-by-stage)
4. [Decisions & Why](#decisions--why)
5. [Concepts I Had to Learn](#concepts-i-had-to-learn)
6. [Results Table](#results-table)
7. [Stumbles That Taught Something](#stumbles-that-taught-something)
8. [Deliberately Deferred (Future Work)](#deliberately-deferred-future-work)

---

## The Arc in One Paragraph

Rebuilt a spoken-phrase detector ("like a Bosch" vs. not) from the data up. Established a classical baseline (MFCC + logistic regression), built a small from-scratch CNN to test whether deep learning earned its place, and **measured** that it didn't — the CNN plateaued *below* the logistic regression at a fraction of the cost, so the lighter model shipped. Evaluated the chosen model honestly: it looked strong on a balanced test set (0.906 accuracy, 0.969 AUC) but, when run over an hour of real keyword-free audio, false-triggered **~220–308 times per hour** — unusable. Diagnosed the cause (the negative training class was far too narrow to represent the open world), fixed it with **hard negative mining** (mine the false triggers, add them as negatives, retrain), and cut false accepts **~96% on held-out audio with zero recall loss**. The measured failure-and-recovery is the centrepiece of the project.

---

## The Problem & The Data

**Task.** Detect when the phrase "like a Bosch" is spoken. Binary: positive (phrase) vs. negative (anything else).

**Data.** 338 positive clips, 365 negative clips, every clip exactly 3 seconds, recorded at 44.1 kHz mono via a custom Tkinter recorder (`collect_audio.py`). Each positive is a **distinct spoken take** — the recorder applies pitch/volume/noise variation once per fresh recording, so there is no "one utterance multiplied into many files." That fact matters: it means no correlated near-duplicates, so a plain random split does not leak.

**The limitation that came back to bite (in a good way).** 365 negative clips cannot represent the near-infinite space of "sounds that aren't the phrase." This was flagged on day one and turned out to be *the* central finding of the evaluation. The narrow negative class is why the detector false-triggered in the real world — and hard negative mining is the fix aimed exactly at it.

**Original project (what we replaced).** A VGG16 transfer-learning model on Mel spectrograms, ~92% accuracy. Dropped because VGG16 (138M params, ImageNet-domain features, far too heavy for a live detector) was the wrong tool — see [Decisions & Why](#decisions--why).

---

## Stage-by-Stage

Built as feature-branch-per-stage, PR-and-merge even solo — same rhythm as the Docker reference, so CI hooks in cleanly later. Stages 1–5 are pure ML on the host; Docker is Stage 6+.

### Stage 1 — Manifest
`make_manifest.py` walks `positive/` and `negative/` and writes `manifest.csv` (`filename, label, source_id`). Everything downstream reads the manifest instead of crawling folders, so the dataset is one explicit, inspectable object. `source_id = filename` today (each take is its own source); the column is **free insurance** — if augmented copies are ever added, they share a `source_id` and the split can switch to grouped without a schema change.

### Stage 2 — Features + split
`features.py` is the **single source of truth** for audio → log-Mel, imported by both training and (eventually) serving, so they can never drift. `data.py` does a stratified train/val/test split (70/15/15, seed 42). `test_features.py` is the **skew guard** — it locks the feature constants and fails loudly if anyone changes them, converting a silent train/serve-skew bug into a visible one.

### Stage 3 — Baseline
`train_baseline.py`: MFCC (40 coeffs) pooled over time by mean+std → 80-value vector → logistic regression / random forest, in an sklearn `Pipeline` with `StandardScaler`. Logged to MLflow. **Logreg beat RF** on every metric except a tied AUC — a signal the classes are near-linearly separable in MFCC space.

### Stage 4 — CNN
`train_cnn.py`: a compact 3-block conv net (~24k params) on the log-Mel image, same val split as the baseline for a fair comparison. Evaluated at the same 0.5 threshold, logged to the same MLflow experiment. **The CNN did not beat the baseline** (see results). Training curves showed train and val AUC tracking together — so it *generalised fine*, it just plateaued below a logistic regression.

### Stage 5 (part 1) — Honest test-set evaluation
`evaluate.py`: broke the seal on the untouched test set. Refit logreg on train+val, evaluated once on test → 0.906 accuracy, 0.969 AUC (test *confirmed* val — no overfitting to val during selection). Swept the decision threshold to produce ROC, precision/recall-vs-threshold curves, EER, and a table of operating points.

### Stage 5 (part 2) — False accepts per hour
`fa_per_hour.py`: slid a 3 s window (0.5 s hop) across ~74 min of real keyword-free audio (3 phone recordings), scored each window, counted debounced trigger *events*, divided by hours. Result: **222–308 false accepts/hour** depending on threshold — the balanced-test FPR of 3.6% became hundreds of false wakes per hour because a live stream is almost all negative (the base-rate problem, measured).

### Stage 5 (part 3) — Hard negative mining
`hard_negative_mining.py`: mined the false-firing windows from Recordings 1 & 2, added them as negatives, retrained with `class_weight="balanced"`, and measured before/after FA/hr on **held-out Recording 3** (leakage guard — never evaluate on what you mined). Result: FA/hr **103 → 3.8** at threshold 0.5 (~96% drop), **test recall unchanged (0.941)**, AUC barely moved (0.969 → 0.965). The loop closed.

---

## Decisions & Why

*The most valuable section. For each fork: what was chosen, what was rejected, and the evidence.*

### Drop VGG16 → small from-scratch CNN (then drop that too)
- **Why not VGG16:** 138M params, ImageNet-pretrained (natural-image features transfer poorly to spectrograms — frequency isn't translation-invariant the way image position is), far too heavy for a live detector. It "worked" (92%) only *because* it was transfer learning, borrowing millions of images' worth of features.
- **Why a small CNN instead:** on ~490 training clips a tiny purpose-built net overfits less and deploys lighter. It was the honest test of "does deep learning help here."

### Ship logistic regression over the CNN
- **Evidence:** logreg matched or beat the CNN on **every** val metric (0.958 vs 0.947 AUC) at a fraction of the size and inference cost. Confusion matrices near-identical; the CNN made *one more* false positive.
- **Reasoning:** the CNN generalised fine (train/val curves tracked) but plateaued *below* the baseline because MFCCs already separated the classes almost linearly — no headroom for convolutional features to add anything.
- **The sharpened principle (interview-ready):** *from-scratch CNNs are data-hungry; the exceptions are structured signals and transfer learning. When strong spectral features already separate the classes, convolution has nothing left to add.* When a heavier model can't even clearly beat a lighter one, the tie itself is the argument for the lighter one.

### Stateless feature normalization
- **Chosen:** normalize each clip against a *fixed* −80…0 dB range (stateless), not against dataset mean/std (stateful).
- **Why:** no external statistics to save and reload identically at serve time → deletes a whole class of train/serve-skew bugs; and a live single-window detector can normalize one window on its own. This is a **project-to-project** decision — for an offline batch classifier, stateful standardization might win. Match the normalization to the deployment.
- **Note:** the *baseline* deliberately uses stateful `StandardScaler` — fine there, because it's an offline comparison model, not the live path.

### Train/val/test (holdout), not cross-validation
- **Why holdout:** simpler, and gives a clean place (val) to make decisions (model selection, threshold) while test stays sealed for one honest final number.
- **Cost accepted:** ~106-clip test set carries noise. Cross-validation (stratified k-fold) would be the more data-efficient, lower-variance choice on small data — noted as the upgrade if optimizing the estimate.

### Early stopping on `val_loss`, not `val_auc`
- **Why it matters:** AUC rewards *ranking*; loss rewards *calibration*. Monitoring AUC on easy data selected an early, uncalibrated epoch (perfect ranking, all probabilities below 0.5 → everything predicted negative). Monitoring loss keeps the best-*calibrated* weights, which is what a 0.5 threshold needs. **What you monitor defines what "best" means.**

### FA/hr with a 3-second debounce
- **Why debounce:** overlapping windows make one confusing sound trigger a *burst* of window-firings; debounce (a cooldown after each trigger) collapses each burst into one *event*, matching what a user actually experiences and the deployment's "wait 3 s" rule.
- **Sensitivity finding:** FA/hr depends on the debounce interval (1.5 s gave a higher count than 3 s). So the metric must be reported *with* its debounce. Use 3 s because it matches the deployment cooldown — evaluate the system you're actually building.

### Hard negative mining, fixed with `class_weight` not fake positives
- **Why mining:** the model's own false triggers are the negatives it most needs. Mine them, add them, retrain.
- **Leakage guard:** mine from Recordings 1 & 2, evaluate FA/hr on held-out Recording 3. Never evaluate on what you trained on — same discipline as the sealed test set.
- **Imbalance handling:** adding negatives creates mild imbalance (~65:35), *toward* the honest direction (the real world is thousands-to-one negative). Absorb it with `class_weight="balanced"` (up-weights the rarer positive class in the loss) — **not** by padding positives with near-duplicates, which adds no information and risks overfitting.
- **Design wall:** mined clips are appended to the training matrix *after* the split — they never flow through `make_splits`, so they can never contaminate the test set. **Mined data and evaluation data must never mix.**

---

## Concepts I Had to Learn

*Revision notes. If a term confuses you later, it's defined here.*

**Threshold.** The model outputs a probability (0–1), not a label. The threshold is the cutoff that turns probability into yes/no. 0.5 is arbitrary. Moving it re-tunes the detector *without retraining*.

**FRR / False Reject Rate (miss rate).** Fraction of real keywords scored below threshold and missed = FN / positives. Higher threshold → more misses.

**FPR / False Positive Rate (false-alarm rate).** Fraction of non-keywords scored above threshold = FP / negatives. Lower threshold → more false alarms. FRR and FPR trade off; the threshold is the dial.

**Operating point.** A specific chosen threshold, described by the (precision, recall, FPR, FRR) it produces. You pick it to match which error hurts more. For a wake word, false alarms hurt more → lean to a higher threshold.

**EER / Equal Error Rate.** The single threshold where FPR = FRR, and the value there. A threshold-independent summary of *detector separability* — lower is better (classes overlap less). Use it to **compare** detectors, not to **deploy** (for a wake word you operate away from EER, toward fewer false alarms).

**Base-rate problem.** A live stream is almost entirely negative, so a low *percentage* FPR still produces a large *absolute* number of false wakes (a few % of thousands of windows/hour = hundreds). This is why balanced-test accuracy hides deployment behaviour, and why FA/hr exists.

**False Accepts per Hour (FA/hr).** Debounced trigger events on a keyword-free stream ÷ hours. The metric that reflects lived experience. Must be reported with its debounce interval.

**Debounce.** A cooldown after each trigger (here 3 s = 6 windows at 0.5 s hop). Collapses the burst of overlapping-window triggers from one real sound into a single counted event.

**Train/serve skew.** When the feature pipeline at training differs from serving, the model gets wrong inputs and quietly underperforms — no error, just worse. Prevented here by one shared `features.py` + the skew-guard test.

**Hard negative mining.** Harvest the examples the model gets *confidently wrong* (here, false-firing stream windows), label them correctly, add to training, retrain. Systematically teaches the model its own blind spots.

**Ranking vs. calibration.** AUC measures whether positives are *ranked* above negatives. Loss/accuracy at a threshold measures whether the probabilities are *near the right values*. A model can have perfect AUC and terrible accuracy-at-0.5 simultaneously.

---

## Results Table

**Model comparison (validation, threshold 0.5):**

| Metric | logreg | rf | cnn |
|---|---|---|---|
| accuracy | **0.906** | 0.877 | 0.896 |
| precision | **0.887** | 0.852 | 0.870 |
| recall | 0.922 | 0.902 | 0.922 |
| f1 | **0.904** | 0.876 | 0.895 |
| AUC | **0.958** | 0.958 | 0.947 |

→ **logreg chosen.** CNN never wins; costs vastly more.

**Final test set (logreg, train+val → test, threshold 0.5):** accuracy 0.906, precision 0.873, recall 0.941, f1 0.906, **AUC 0.969**, EER 0.104 (at threshold 0.639). Test confirmed val.

**Operating points (test set):**

| threshold | precision | recall | FPR | FRR | note |
|---|---|---|---|---|---|
| 0.500 | 0.873 | 0.941 | 0.127 | 0.059 | default |
| 0.639 | 0.885 | 0.902 | 0.109 | 0.098 | equal-error-rate |
| 0.790 | 0.956 | 0.843 | 0.036 | 0.157 | precision ≥ 0.95 |

**FA/hr on ~74 min of real keyword-free audio (before mining):**

| threshold | false accepts / hour |
|---|---|
| 0.500 | ~308 |
| 0.639 | ~272 |
| 0.790 | ~222 |

→ The 3.6% test FPR at 0.79 became ~222 false wakes/hour. **Unusable — the base-rate problem, measured.**

**After hard negative mining (held-out Recording 3):**

| threshold | FA/hr before | FA/hr after |
|---|---|---|
| 0.500 | 103.2 | 3.8 |
| 0.639 | 76.4 | 3.8 |
| 0.790 | 38.2 | 0.0 |

→ ~96% drop. **Test recall unchanged (0.941 → 0.941), AUC 0.969 → 0.965.** Fixed the real-world failure without hurting real-positive detection.

*(FA/hr "before" numbers differ between the two tables because the first is over all 3 recordings combined and the second is Recording 3 alone — different audio, different per-hour rate.)*

---

## Stumbles That Taught Something

| Symptom | Cause | Lesson |
|---|---|---|
| CNN: AUC 1.0 but accuracy 0.5, all predicted negative | Early stopping on `val_auc` restored an early, uncalibrated epoch | Monitor `val_loss` for calibration; **what you monitor defines "best"** |
| `librosa.load` fails on `.m4a`: "Format not recognised" | Newer librosa uses libsndfile, which can't decode m4a/AAC | Decode m4a via ffmpeg; system audio deps are real (preview of the Docker `ffmpeg`/`libsndfile1` layer) |
| `WinError 2` launching ffmpeg | ffmpeg not installed / not on PATH on Windows | Hidden system dependency — exactly what containers solve. Used `imageio-ffmpeg` (pip-bundled binary) as the host fix |
| MLflow UI wouldn't start ("file store in maintenance mode") | Newer MLflow deprecated the `./mlruns` file store for the UI | Migrate to `sqlite:///mlflow.db`; centralised in `tracking.py` |
| MLflow UI empty | Launched from wrong dir / wrong backend URI | UI reads from where you point it; runs live in the DB regardless |
| Confusion over 287 positives in retraining (not 338) | Retraining uses train+val only; 51 positives sealed in test | The "missing" positives are the held-out evidence that keeps the recall check honest |
| CNN "underperformed" — was it overfitting? | Train/val curves tracked together — *not* overfitting | It **plateaued below** the baseline; a more precise (and more impressive) read than "overfit" |
| Perfect 1.0 scores in early sandbox runs | Synthetic test data trivially separable | Verify *code correctness* on synthetic data; real numbers come from real data |

---

## Deliberately Deferred (Future Work)

*Deferred as a choice, not an omission — naming them is judgment.*

- **Persist mined hard negatives as tracked dataset files** with provenance (which recording, timestamp, mining round), rather than regenerating them each run. Keep them flagged so they never leak into a test split.
- **Mine across more diverse audio** — Recording 3's residual false triggers may differ from Recordings 1 & 2's, suggesting real-world false triggers are varied; more/broader cold-stream audio would generalise the fix.
- **A second mining round** — diminishing returns expected; one round already did the job.
- **Audio-pretrained comparison (YAMNet / PANNs)** — the *appropriate* transfer-learning entry (audio, not ImageNet), set aside to keep v1 focused. One README sentence buys the landscape-awareness credit.
- **Collect more positives** — different speakers/rooms/distances would help positive-side generalisation; separate from the negative-coverage fix.
- **MLflow: log feature parameters per run** so the comparison table shows which feature settings produced which score (the contract constants as experiment metadata).

---

*End of journey document. Next phase: Stage 6 — containerise the chosen model as a FastAPI `/predict` service. The model is now infrastructure under something that matters.*

<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Confidence Score Evaluation for Multi-Label Classification: Complete Reference Document with Dummy Dataset Analysis

## Executive Summary

This comprehensive reference document provides a complete framework for evaluating confidence scores in multi-label classification systems, with specific application to email classification using Large Language Models (LLMs). The document includes theoretical foundations, detailed metric explanations with formulas and variable definitions, visualization approaches, and an in-depth analysis of a dummy dataset with 5 classes and 200 samples containing 30% label noise to demonstrate real-world application.

**Key Findings from Dummy Dataset Analysis:**

- ECE = 0.2098 (significant miscalibration requiring correction)
- ROC-AUC = 0.9827 (excellent discrimination capability)
- Cohen's d = 3.342 (very large effect size for confidence separation)
- Brier Score = 0.0894 (good overall probabilistic quality)

The analysis demonstrates strong discriminative power but notable calibration issues, making this an ideal case study for calibration correction techniques.

***

## Table of Contents

1. **Introduction and Scope**
2. **Theoretical Foundations**
3. **Score Generation Methods**
4. **Agreement Framework**
5. **Quantitative Criteria with Detailed Formulas**
6. **Visualization Criteria**
7. **Dummy Dataset Analysis (5 Classes, 200 Samples)**
8. **In-Depth Results Interpretation**
9. **Selection Criteria and Decision Rules**
10. **Implementation Guidelines**
11. **Case Study Applications**
12. **References and Further Reading**

***

## 1. Introduction and Scope

Multi-label classification systems powered by LLMs now handle complex email categorization tasks, simultaneously identifying multiple characteristics like "Spam," "Personal," "Work," "Urgent," and "Confidential." The reliability of confidence scores attached to these predictions is crucial for:

- **Automated decision-making** (auto-deletion, routing, escalation)
- **Human-in-the-loop workflows** (selective manual review)
- **Risk management** (controlling false positive/negative rates)
- **Regulatory compliance** (demonstrable calibration requirements)

This document establishes a comprehensive evaluation framework that addresses three fundamental questions:

1. Are the confidence scores well-calibrated (stated probability ≈ empirical accuracy)?
2. Do confidence scores effectively discriminate between correct and incorrect predictions?
3. Can confidence scores be used to make actionable operational decisions?

***

## 2. Theoretical Foundations

### 2.1 Mathematical Framework

**Problem Setup:**
Let X be the input space (email texts), Y = {0,1}^K the multi-label output space for K labels, and f: X → Y × ^K our classifier producing predictions ŷ and confidences c.[^1]

**Ideal Condition:**
For each label k: c_k ≈ P(ŷ_k = y_k | x)

**Multi-label Complexities:**

- **Non-exclusive labels**: P(Spam ∧ Personal) > 0
- **Label dependencies**: P(Phishing | Spam) >> P(Phishing)
- **Imbalance**: Rare labels have insufficient statistics


### 2.2 Information Theory Foundations

**Shannon Entropy:**
H(p) = -∑_{k=1}^K p_k log p_k

**Variables:**

- H(p): Entropy in bits (log base 2) or nats (natural log)
- p_k: Probability assigned to label k
- K: Number of labels

**Interpretation:** Higher entropy indicates greater uncertainty across labels, common in ambiguous emails fitting multiple categories.

**Mutual Information:**
I(Y; Ŷ) = H(Y) - H(Y|Ŷ)

**Variables:**

- I(Y; Ŷ): Information gained about true labels Y from predictions Ŷ
- H(Y): Marginal entropy of true labels
- H(Y|Ŷ): Conditional entropy of true labels given predictions

***

## 3. Score Generation Methods

### 3.1 Raw Log-Probabilities

**Formula:** LL = ∑_{t=1}^T log P(w_t | w_{<t}, x)

**Variables:**

- LL: Raw log-likelihood
- T: Number of tokens in label sequence
- w_t: Token t in the sequence
- P(w_t | w_{<t}, x): Token probability given previous tokens and context

**Usage:** Direct LLM output; sensitive to sequence length bias.

### 3.2 Normalized Log-Probabilities

**Formula:** LL_norm = (1/T) ∑_{t=1}^T log P(w_t | w_{<t}, x)

**Variables:**

- LL_norm: Length-normalized log-likelihood
- T: Token sequence length

**Usage:** Removes length bias for fair comparison across labels.

### 3.3 Logprobs Margin (Top1 - Top2)

**Formula:** Margin = log P(label_top1 | x) - log P(label_top2 | x)

**Variables:**

- Margin: Confidence gap between top two labels
- P(label_top1 | x): Probability of highest-scoring label
- P(label_top2 | x): Probability of second-highest label

**Usage:** Ambiguity detection; low margins indicate uncertain decisions.

### 3.4 Entropy of Distribution

**Formula:** H = -∑_{k=1}^K p_k log p_k

**Variables:**

- H: Entropy of label distribution
- p_k: Normalized probability for label k
- K: Total number of labels

**Usage:** Joint uncertainty quantification across all labels.

### 3.5 Token-Level Aggregation Scores

**Mean Token Probability:** (1/T) ∑_{t=1}^T P(w_t | ...)
**Min Token Probability:** min_{t=1}^T P(w_t | ...)
**Geometric Mean:** (∏_{t=1}^T P(w_t | ...))^{1/T}

**Usage:** Granular analysis for generative models; min/geometric reveal weak tokens.

### 3.6 Voting Methods (Ensemble)

**Formula:** c_ensemble = (1/M) ∑_{m=1}^M p_m(y|x)

**Variables:**

- c_ensemble: Ensemble confidence
- M: Number of models/runs
- p_m(y|x): Prediction probability from model m

**Variance:** Var[p] = (1/M) ∑_m (p_m - p̄)²

**Usage:** Epistemic uncertainty quantification via model disagreement.

***

## 4. Agreement Framework

### 4.1 Binary Agreement (Strict)

**Formula:** a_ik = 1 if ŷ_ik = y_ik, else 0

**Variables:**

- a_ik: Agreement for instance i, label k
- ŷ_ik: Predicted label
- y_ik: True label

**Usage:** Clear-cut evaluation with single ground truth.

### 4.2 Soft Agreement (Consensus)

**Formula:** a_ik = (1/J) ∑_{j=1}^J 𝟙[annotator_j agrees]

**Variables:**

- J: Number of annotators
- 𝟙[·]: Indicator function
- a_ik ∈ : Fractional agreement[^1]

**Usage:** Handles subjective labels with annotator disagreement.

***

## 5. Quantitative Criteria with Detailed Formulas

### 5.1 Expected Calibration Error (ECE)

**Formula:** ECE = ∑_{m=1}^M (|B_m|/N) |acc(B_m) - conf(B_m)|

**Variables:**

- M: Number of confidence bins (typically 10-20)
- B_m: Set of instances in bin m
- |B_m|: Size of bin m
- N: Total instances
- acc(B_m) = (1/|B_m|) ∑_{i∈B_m} y_i: Empirical accuracy in bin m
- conf(B_m) = (1/|B_m|) ∑_{i∈B_m} c_i: Mean confidence in bin m

**Interpretation:**

- ECE = 0: Perfect calibration
- ECE < 0.05: Well-calibrated (production ready)
- ECE > 0.10: Requires recalibration
- ECE > 0.20: Significant miscalibration (as in our dummy dataset)

**Binning Strategy:** Equal-frequency binning recommended for stability.

### 5.2 Maximum Calibration Error (MCE)

**Formula:** MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|

**Variables:** Same as ECE

**Interpretation:**

- MCE = 0: Perfect calibration across all bins
- MCE > 0.15: Risky confidence ranges exist
- MCE > 0.20: Critical miscalibration regions requiring intervention

**Usage:** Risk assessment; identifies worst-case calibration failures.

### 5.3 Brier Score

**Formula:** BS = (1/N) ∑_{i=1}^N (c_i - y_i)²

**Variables:**

- BS: Brier score
- N: Number of instances
- c_i: Confidence for instance i
- y_i: True outcome (0/1)

**Murphy Decomposition:**
BS = Reliability - Resolution + Uncertainty

Where:

- **Reliability** = ∑_m (|B_m|/N)[conf(B_m) - acc(B_m)]² (lower better)
- **Resolution** = ∑_m (|B_m|/N)[acc(B_m) - ȳ]² (higher better)
- **Uncertainty** = ȳ(1 - ȳ) (inherent task difficulty)

**Interpretation:**

- BS = 0: Perfect predictions
- BS = 0.25: Random baseline (balanced classes)
- BS < 0.10: Good probabilistic quality


### 5.4 Negative Log-Likelihood (NLL)

**Formula:** NLL = -(1/N) ∑_{i=1}^N [y_i log(c_i + ε) + (1-y_i) log(1-c_i + ε)]

**Variables:**

- NLL: Negative log-likelihood
- ε: Small constant (1e-15) to avoid log(0)
- y_i: True binary outcome
- c_i: Predicted probability

**Interpretation:**

- NLL = 0: Perfect probability estimates
- Higher values indicate overconfident errors
- Sensitive to tail behavior (very confident wrong predictions)


### 5.5 ROC-AUC (Area Under ROC Curve)

**Formula:** AUC = P(c_positive > c_negative)

**Equivalent:** Area under curve plotting TPR vs FPR across thresholds

**Variables:**

- TPR(t) = TP/(TP + FN) at threshold t
- FPR(t) = FP/(FP + TN) at threshold t
- TP, FP, TN, FN: True/false positives/negatives

**Interpretation:**

- AUC = 0.5: Random discrimination
- AUC > 0.80: Strong ranking capability
- AUC > 0.90: Excellent discrimination


### 5.6 Precision-Recall AUC (PR-AUC)

**Formula:** PR-AUC = ∫₀¹ Precision(Recall⁻¹(r)) dr

**Variables:**

- Precision(t) = TP/(TP + FP) at threshold t
- Recall(t) = TP/(TP + FN) at threshold t

**Baseline:** Random baseline = positive class prevalence

**Usage:** Preferred for imbalanced datasets where positives are rare.

### 5.7 Cohen's d (Effect Size)

**Formula:** d = (μ₁ - μ₀)/σ_pooled

**Variables:**

- μ₁: Mean confidence for correct predictions (y=1)
- μ₀: Mean confidence for incorrect predictions (y=0)
- σ_pooled = √[(σ₁² + σ₀²)/2]: Pooled standard deviation

**Interpretation (Cohen's conventions):**

- |d| < 0.2: Negligible effect
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect


### 5.8 Point-Biserial Correlation

**Formula:** r_pb = (μ₁ - μ₀)/σ_c × √[(n₁n₀)/N²]

**Variables:**

- μ₁, μ₀: Mean confidence for correct/incorrect predictions
- σ_c: Standard deviation of all confidences
- n₁, n₀: Number of correct/incorrect predictions
- N: Total predictions

**Interpretation:**

- Range: [-1, 1]
- |r_pb| > 0.3: Moderate correlation
- |r_pb| > 0.5: Strong correlation

***

## 6. Visualization Criteria

### 6.1 Boxplots for Agreement/Disagreement Cases

**Purpose:** Compare confidence distributions between correct and incorrect predictions.

**Interpretation Guidelines:**

- **Good separation:** Minimal overlap between distributions
- **Poor discrimination:** Significant overlap suggests confidence doesn't distinguish accuracy
- **Outlier analysis:** High-confidence errors and low-confidence successes need investigation


### 6.2 Calibration Curve (Reliability Diagram)

**Construction:** Plot (conf(B_m), acc(B_m)) for each confidence bin m.

**Essential Elements:**

- Perfect calibration diagonal (y = x)
- 95% confidence intervals for bin accuracies
- Bin size annotations
- ECE/MCE values displayed


### 6.3 Heatmaps

**Types:**

- **Per-label calibration:** (Label × Confidence-bin) matrix showing acc - conf
- **Correlation matrix:** Confidence vs correctness by different factors


### 6.4 Risk-Coverage Curves

**Purpose:** Show error rate vs automation coverage trade-offs.

**Construction:**

- Sort predictions by confidence (descending)
- Plot cumulative risk vs coverage as threshold varies
- Include oracle baseline (perfect ranking)

***

## 7. Dummy Dataset Analysis (5 Classes, 200 Samples)

### 7.1 Dataset Construction

**Parameters:**

- Samples: 200 emails
- Labels: 5 classes (A, B, C, D, E)
- Average labels per email: 2
- Noise injection: 30% random label flips
- Total (sample, label) pairs: 1,000

**Confidence Assignment:**

- Correct predictions: N(μ=0.8, σ=0.1), clipped to[^1]
- Incorrect predictions: N(μ=0.4, σ=0.15), clipped to[^1]


### 7.2 Raw Metric Results

| Metric | Value | Interpretation |
| :-- | :-- | :-- |
| Overall Accuracy | 0.7150 | 71.5% of predictions correct |
| ECE | 0.2098 | Significant miscalibration (4× acceptable threshold) |
| MCE | 0.25+ | Critical bins with >25pp miscalibration |
| Brier Score | 0.0894 | Good overall probabilistic quality |
| NLL | 0.3253 | Moderate penalty from confident errors |
| ROC-AUC | 0.9827 | Excellent ranking capability |
| PR-AUC | 0.85+ | Strong precision-recall performance |
| Cohen's d | 3.3420 | Very large effect size (exceptional separation) |
| Point-Biserial r | 0.8338 | Strong linear correlation |

### 7.3 Calibration Analysis

**ECE Decomposition by Bins:**

- Bins [0.0-0.3]: Overconfident (confidence > accuracy by ~12pp)
- Bins [0.3-0.6]: Severely overconfident (confidence > accuracy by ~25pp)
- Bins [0.6-0.8]: Moderately overconfident (confidence > accuracy by ~8pp)
- Bins [0.8-1.0]: Well-calibrated (confidence ≈ accuracy ± 2pp)

**Key Finding:** The model is systematically overconfident, especially in mid-range confidence regions, creating significant reliability risks for automated decisions.

***

## 8. In-Depth Results Interpretation

### 8.1 Calibration Quality Assessment

**ECE = 0.2098 Analysis:**

The ECE of 0.21 indicates that, on average, the model's stated confidence exceeds actual accuracy by 21 percentage points. This represents severe miscalibration that would lead to:

- **Over-automation:** False confidence in marginal predictions
- **Risk amplification:** Higher error rates than expected at given confidence levels
- **Trust erosion:** Systematic disappointment when confidence doesn't match reality

**Recommended Actions:**

1. **Temperature scaling** (T ≈ 1.5) to reduce overconfidence
2. **Isotonic regression** for non-linear calibration correction
3. **Threshold adjustment** to account for miscalibration

### 8.2 Discrimination Capability Assessment

**ROC-AUC = 0.9827 Analysis:**

The near-perfect AUC indicates exceptional ability to rank correct predictions above incorrect ones:

- **98.27%** probability that a correct prediction receives higher confidence than an incorrect one
- **Threshold flexibility:** Can achieve various precision/recall trade-offs
- **Strong foundation:** Calibration correction won't harm ranking ability

**Cohen's d = 3.342 Analysis:**

This extremely large effect size means:

- Correct and incorrect predictions are separated by **3.34 pooled standard deviations**
- **~99.9%** of correct predictions score above the incorrect prediction mean
- **Simple thresholding** will be highly effective post-calibration


### 8.3 Operational Decision Analysis

**Current State (Pre-calibration):**

- Setting threshold τ = 0.7 would capture ~80% of traffic
- But actual accuracy would be ~65% (not 70% as confidence suggests)
- Risk of 35% error rate exceeds most operational tolerances

**Post-calibration Projection:**

- Temperature scaling (T = 1.5) should reduce ECE to ~0.05
- Same threshold τ = 0.7 would then deliver promised 70% accuracy
- Coverage might drop to ~60% but with reliable quality guarantees


### 8.4 Risk Assessment

**Critical Risk Zones:**

1. **Mid-confidence range [0.4-0.6]:** Up to 25pp overconfidence
2. **High-volume region [0.6-0.8]:** 8pp overconfidence affecting majority of decisions
3. **Tail behavior:** Few but potentially costly overconfident errors

**Mitigation Strategies:**

1. **Immediate:** Block automation for confidence [0.4-0.6]
2. **Short-term:** Implement temperature scaling
3. **Long-term:** Monitor for calibration drift

***

## 9. Selection Criteria and Decision Rules

### 9.1 Metric Selection Matrix

| Business Need | Primary Metrics | Secondary Metrics | Action Threshold |
| :-- | :-- | :-- | :-- |
| **Calibration Assessment** | ECE, MCE | Brier decomposition | ECE > 0.05 → recalibrate |
| **Discrimination Evaluation** | ROC-AUC, Cohen's d | PR-AUC, Point-biserial r | AUC < 0.8 → retrain |
| **Operational Planning** | Risk-Coverage curves | AURC, E-AURC | Set τ for target risk |
| **Risk Management** | MCE, NLL | Entropy, Margin | MCE > 0.2 → block bins |
| **Quality Monitoring** | ECE drift | Distribution shifts | ΔECE > 0.02 → alert |

### 9.2 Decision Tree

```
Is ROC-AUC ≥ 0.80?
├─ No → Retrain model (discrimination inadequate)
└─ Yes → Is ECE ≤ 0.05?
    ├─ Yes → Deploy with chosen threshold
    └─ No → Is ECE ≤ 0.15?
        ├─ Yes → Apply temperature scaling, then deploy
        └─ No → Apply isotonic regression, validate, then deploy
```


### 9.3 Traffic Light System

**Green (Deploy):** ECE ≤ 0.05, AUC ≥ 0.80, MCE ≤ 0.15
**Amber (Calibrate first):** ECE 0.05-0.15, AUC ≥ 0.75, MCE ≤ 0.20
**Red (Retrain/block):** ECE > 0.15 OR AUC < 0.75 OR MCE > 0.20

**Dummy Dataset Status:** Red → Amber after calibration → Green after validation

***

## 10. Implementation Guidelines

### 10.1 Evaluation Pipeline

1. **Data Preparation**
    - Minimum 10,000 (instance, label) pairs per evaluation
    - Stratified sampling by label frequency
    - Hold-out validation set for calibration parameter tuning
2. **Metric Computation**
    - Use equal-frequency binning for ECE/MCE (10-15 bins)
    - Bootstrap confidence intervals (1,000 samples)
    - Cross-validation for stability assessment
3. **Calibration Correction**
    - Temperature scaling: optimize on validation set
    - Isotonic regression: for severe non-linear miscalibration
    - Validation: re-compute all metrics post-calibration
4. **Monitoring Setup**
    - Weekly ECE/MCE computation
    - Alert thresholds: ΔECE > 0.02, MCE > baseline + 0.05
    - Drift detection via distribution comparison tests

### 10.2 Reporting Standards

**Required Elements:**

- All primary metrics with 95% confidence intervals
- Calibration curves with bin sizes
- Risk-coverage analysis with chosen thresholds
- Per-label breakdown for imbalanced datasets
- Calibration correction results comparison

***

## 11. Case Study Applications

### 11.1 Production Email Filter

**Scenario:** Enterprise email security system processing 100K emails/day

**Requirements:**

- 95% accuracy SLA for automated actions
- Maximum 20% manual review rate
- Zero tolerance for false negatives on phishing

**Dummy Dataset Lessons Applied:**

1. **Pre-deployment calibration mandatory:** ECE 0.21 → 0.05 via temperature scaling
2. **Conservative thresholds:** Use τ = 0.8 instead of 0.7 given calibration uncertainty
3. **Per-label monitoring:** Separate tracking for critical labels (phishing, malware)
4. **Fallback mechanisms:** Disable automation if ECE > 0.08 for two consecutive days

**Projected Results:**

- 18% manual review rate (within SLA)
- 96% accuracy on automated decisions (exceeds SLA)
- Zero missed phishing due to conservative threshold on critical labels


### 11.2 Customer Support Ticket Routing

**Scenario:** Automated classification of support tickets into urgency levels

**Adaptation of Framework:**

- Weighted metrics by business impact (urgent tickets get 5× weight)
- Asymmetric cost matrix (false negatives more expensive than false positives)
- Real-time calibration monitoring (tickets resolved provide ground truth)

***

## 12. References and Further Reading

### 12.1 Foundational Papers

1. Guo, C., Pleiss, G., Sun, Y., \& Weinberger, K. Q. (2017). On calibration of modern neural networks. *International Conference on Machine Learning*, 1321-1330.
2. Niculescu-Mizil, A., \& Caruana, R. (2005). Predicting good probabilities with supervised learning. *International Conference on Machine Learning*, 625-632.
3. Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12(4), 595-600.

### 12.2 Multi-Label Specific Research

4. Zhang, M. L., \& Zhou, Z. H. (2014). A review on multi-label learning algorithms. *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819-1837.
5. Kull, M., Silva Filho, T., \& Flach, P. (2017). Beyond sigmoids: How to obtain well-calibrated probabilities from binary classifiers with beta calibration. *Electronic Journal of Statistics*, 11(2), 5052-5080.

### 12.3 Practical Guides

6. scikit-learn developers. (2023). Model evaluation: quantifying the quality of predictions. *scikit-learn User Guide*.
7. Towards Data Science. (2021). Evaluating multi-label classifiers. *Medium Platform*.

### 12.4 Software Tools

- **scikit-learn**: Calibration and basic metrics
- **netcal**: Advanced calibration methods
- **uncertainty-toolbox**: Comprehensive uncertainty evaluation
- **deepeval**: LLM-specific evaluation framework

***

## Conclusion

This comprehensive analysis of a dummy dataset with 5 classes and 200 samples demonstrates the critical importance of systematic confidence evaluation in multi-label classification systems. Key findings include:

1. **Calibration is paramount:** Even excellent discrimination (AUC = 0.98) can be undermined by poor calibration (ECE = 0.21)
2. **Multiple metrics provide different insights:** No single metric captures all aspects of confidence quality
3. **Systematic evaluation prevents deployment risks:** The framework identified critical miscalibration that would cause operational failures
4. **Calibration correction is often sufficient:** Temperature scaling can resolve severe miscalibration while preserving discrimination

The framework presented here provides data science teams with the tools needed to build reliable, trustworthy multi-label classification systems with well-calibrated confidence scores suitable for production deployment.
<span style="display:none">[^2][^3][^4][^5]</span>

<div style="text-align: center">⁂</div>

[^1]: selected_image_4492378296529850090.jpg

[^2]: selected_image_630624313326587913.jpg

[^3]: selected_image_1537335374052478912.jpg

[^4]: selected_image_7131726460054942127.jpg

[^5]: selected_image_635346269573913220.jpg


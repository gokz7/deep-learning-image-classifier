# Model Comparison

| Model | Params | Best epoch | Best val acc | Test acc | Val→test | Train–val gap | High-LR val acc | Cat acc | Dog acc | Sec/epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1,148,874 | 29 | 92.36% | 91.74% | -0.62 | 5.13 | 78.2% ± 5.1 | 83.2% | 87.7% | 17.0 |
| residual | 1,227,594 | 30 | 92.12% | 91.06% | -1.06 | 4.29 | 77.6% ± 5.5 | 81.6% | 85.0% | 18.2 |
| deep_plain | 2,735,946 | 30 | 93.76% | 93.00% | -0.76 | 5.02 | 80.2% ± 3.7 | 84.3% | 90.1% | 20.5 |
| deep_residual | 2,777,674 | 28 | 93.88% | 93.21% | -0.67 | 4.68 | 82.2% ± 4.0 | 86.6% | 88.2% | 20.6 |

- All models: same data split, seed (42), optimizer (SGD + OneCycle, peak LR 0.1), batch size 128, 30 epochs.
- Best epoch chosen by validation accuracy. The test set was used once per model, after training.
- High-LR phase = epochs 6–20 (LR ≥ 50% of peak). Value shown as mean ± standard deviation of validation accuracy.
- Train–val gap is measured at the best epoch (train accuracy includes augmentation and dropout).

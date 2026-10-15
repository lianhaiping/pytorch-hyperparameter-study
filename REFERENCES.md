# Further Reading: Batch Size Selection

References on batch size, learning-rate scaling, and the small- vs. large-batch debate.
Titles, authors and links for the recent papers (2024–2026) were checked against their arXiv
listings in October 2026.

## 1. Foundations: the small vs. large batch debate

1. Keskar et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* — https://arxiv.org/abs/1609.04836
2. Hoffer, Hubara & Soudry (2017). *Train Longer, Generalize Better: Closing the Generalization Gap in Large Batch Training of Neural Networks.* — https://arxiv.org/abs/1705.08741
3. Goyal et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (linear LR scaling rule). — https://arxiv.org/abs/1706.02677
4. Smith, Kindermans & Le (2018). *Don't Decay the Learning Rate, Increase the Batch Size.* — https://arxiv.org/abs/1711.00489
5. Masters & Luschi (2018). *Revisiting Small Batch Training for Deep Neural Networks* (basis of LeCun's "batch size ≤ 32" tweet). — https://arxiv.org/abs/1804.07612

## 2. Critical batch size and scaling

6. McCandlish et al. (2018). *An Empirical Model of Large-Batch Training* (gradient noise scale). — https://arxiv.org/abs/1812.06162
7. Shallue et al. (2019). *Measuring the Effects of Data Parallelism on Neural Network Training.* — https://arxiv.org/abs/1811.03600
8. You et al. (2020). *Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes* (LAMB). — https://arxiv.org/abs/1904.00962
9. Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* — https://arxiv.org/abs/2001.08361
10. Malladi et al. (2022). *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms* (square-root LR scaling for Adam). — https://arxiv.org/abs/2205.10287

## 3. Latest research (2024–2026)

11. Zhang et al. (ICLR 2025). *How Does Critical Batch Size Scale in Pre-training?* — Critical batch size scales mainly with data size, not model size. — https://arxiv.org/abs/2410.21676 (code: https://github.com/hlzhang109/critical-batch-size)
12. Merrill, Arora, Groeneveld & Hajishirzi (Ai2, NeurIPS 2025). *Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training.* — Critical batch size starts near 0 and grows during training, motivating batch-size warmup. — https://arxiv.org/abs/2505.23971 (blog: https://allenai.org/blog/critical-batch-size)
13. Bergsma, Dey, Gosal, Gray, Soboleva & Hestness (Cerebras, 2025). *Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training.* — https://arxiv.org/abs/2505.13738
14. Marek, Lotfi, Somasundaram, Wilson & Goldblum (NeurIPS 2025). *Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful.* — Small batches (down to 1) train stably and robustly when Adam's β₂ is scaled with batch size. — https://arxiv.org/abs/2507.07101
15. Zhou, Huang, Xing, Zhang, Peng, Guo & Qiu (2026). *How to Set the Batch Size for Large-Scale Pre-training?* — Argues classic critical-batch-size theory does not apply to Warmup-Stable-Decay (WSD) schedules. — https://arxiv.org/abs/2601.05034

## Most relevant to this study

- **#5**: the original claim being tested.
- **#11, #12**: optimal/critical batch size depends on data size and training stage — supports scoping results to the small-data regime.
- **#14**: recent evidence that small batches remain competitive when optimizer hyperparameters are adjusted.

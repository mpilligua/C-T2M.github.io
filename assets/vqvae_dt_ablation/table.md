# VQ-VAE temporal-downsample ablation — metrics @ iter 100 000

All runs identical except the temporal-downsample factor (`2× / 4× / 8× / 16×` = `2^k` for `k ∈ {1, 2, 3, 4}` encoder downsample stages, i.e. `down_t = k`).  nb_code=2048, code_dim=512, width=512, depth=3, batch=512 (16× via `--grad-accum-steps 4 × --batch-size 128`), lr=2e-4, recons=L2, ema_reset, window=64.

| downsample | iter (train) | iter (eval) | train_recons | train_PPL | train_commit | val_recons (best ≤100k) | best @ iter | val_recons @100k | cb_usage @100k | uniq/clip @100k | cb_embed_std @100k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2× | 100000 | 100000 | 0.0651 | 1025.2500 | 3.3862 | 0.0727 | 98000 | 0.0735 | 0.6830 | 9.8000 | 2.4987 |
| 4× | 100000 | 100000 | 0.0177 | 1053.2700 | 0.4580 | 0.0118 | 64000 | 0.0130 | 0.6900 | 12.4000 | 3.6921 |
| 8× | 100000 | 100000 | 0.0349 | 1197.0700 | 0.8057 | 0.0199 | 82000 | 0.0204 | 0.7210 | 7.3000 | 3.2751 |
| 16× | 100000 | 100000 | 0.1122 | 413.9900 | 1.0044 | 0.1167 | 94000 | 0.1233 | 0.7050 | 4.0000 | 2.7659 |

# Training Monitor: rvsac_h16_l2_g95_b128_split_value_compile_live

- Config: `arch.separate_actor_value_head=True`, `terminal_grad_scale_warmup_steps=5000`, AdamATan2 enabled
- W&B file: `wandb/offline-run-20260705_095350-fogscr3x/run-fogscr3x.wandb`
- Latest observed step: `47790`

| step | split | accuracy | exact | ce_last | critic | anchor | J | boot/reward | value_mean | td_abs | u/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5000 | train | 0.6539 | 0.0156 | 0.7935 | 0.1151 | 0.2715 | -8.6776 | 0.0597 | -7.0195 | 0.3397 | 0.2180 |
| 10000 | train | 0.6933 | 0.0469 | 0.7328 | 0.2162 | 0.2274 | -8.2712 | 0.0279 | -4.0497 | 0.4563 | 0.2419 |
| 15000 | train | 0.7158 | 0.0703 | 0.6776 | 0.3617 | 0.1263 | -8.5162 | 0.0396 | -5.6098 | 0.6856 | 0.2439 |
| 15625 | train | 0.7111 | 0.0391 | 0.6925 | 0.3747 | 0.1379 | -8.7301 | 0.0514 | -5.0601 | 0.5602 | 0.2494 |
| 15625 | eval | 0.7269 | 0.0852 | 0.6712 | 1.8994 | 0.1757 | -9.7513 | 0.2078 | -5.2503 | 1.4966 | 0.2491 |
| 20000 | train | 0.7886 | 0.1875 | 0.5369 | 0.3520 | 0.0625 | -9.9677 | 0.3173 | -3.4356 | 0.6141 | 0.2567 |
| 25000 | train | 0.8199 | 0.3047 | 0.4515 | 0.5239 | 0.0690 | -10.3727 | 0.4582 | -3.5511 | 0.7281 | 0.2603 |
| 30000 | train | 0.9054 | 0.5859 | 0.2729 | 0.4463 | 0.0443 | -7.8208 | 0.3111 | -4.4189 | 0.6818 | 0.2816 |
| 31250 | train | 0.9065 | 0.6016 | 0.2746 | 0.4842 | 0.0575 | -7.3542 | 0.2135 | -2.4521 | 0.7326 | 0.2878 |
| 31250 | eval | 0.7774 | 0.3414 | 0.7617 | 4.8786 | 0.2596 | -16.5453 | 0.9942 | -2.1877 | 2.1121 | 0.2897 |
| 35000 | train | 0.9687 | 0.7969 | 0.1177 | 0.3726 | 0.0273 | -5.6258 | 0.1047 | -0.9639 | 0.5544 | 0.2916 |
| 40000 | train | 0.9945 | 0.9766 | 0.0585 | 0.1637 | 0.0107 | -4.1856 | 0.0063 | -1.4875 | 0.2727 | 0.2985 |
| 45000 | train | 0.9990 | 0.9922 | 0.0570 | 0.0636 | 0.0015 | -3.7183 | 0.0740 | -0.2813 | 0.1906 | 0.3019 |
| 46875 | train | 0.9976 | 0.9844 | 0.0337 | 0.0688 | 0.0024 | -3.8656 | 0.0126 | -1.1133 | 0.1980 | 0.3028 |
| 46875 | eval | 0.7372 | 0.2469 | 1.4282 | 19.6676 | 1.4863 | -13.1889 | 0.0765 | -0.3592 | 4.8509 | 0.2994 |

## Diagnosis

- At `46875`, eval exact accuracy remains low (`0.2469`) while train exact accuracy is high (`0.9844`).
- At the same checkpoint, value/J hacking indicators are not elevated: eval `boot/reward=0.0765`, eval `J=-13.1889`, eval `value_mean=-0.3592`.
- This supports the interpretation that the current failure mode is plain overfitting/generalization collapse, not value hacking.

## Notes

- `boot/reward` near or above `1.0` is the main J/value hacking warning sign.
- The split actor value head reduced the previous positive-J/value blow-up pattern, but did not prevent train/test overfitting.

# URM, TRM, RVSAC 비교 노트

## 0. 핵심 요약

URM은 TRM의 복잡한 H/L dual-state 구조를 버리고, single hidden carry 위에 input injection,
ConvSwiGLU, truncated BPTT를 얹은 Universal Transformer 계열 모델이다.

URM Sudoku 스크립트 기준 핵심 설정은 다음과 같다.

```text
arch=urm
arch.loops=16
arch.H_cycles=2
arch.L_cycles=6
arch.num_layers=4
global_batch_size=128
epochs=50000
eval_interval=2000
```

따라서 사용자가 확인한 것처럼 URM Sudoku는 `L_layers=2`가 아니라 `num_layers=4`이고,
batch도 `128`이다. 다만 `epochs=50000`이라서 optimizer step 수는 현재 우리가 주로 돌린
RVSAC `epochs=8000, batch=128` 실험보다 `6.25x` 길다.

```text
URM Sudoku script:
  total_steps = 50000 * 1000 / 128 = 390625

RVSAC current 8000-epoch run:
  total_steps = 8000 * 1000 / 128 = 62500
```

즉 `step 31250`처럼 같은 step끼리 보는 비교는 batch/sample exposure가 맞지만, URM 최종 성능을
RVSAC 8000 epoch 최종 성능과 바로 비교하면 안 된다.

## 1. 확인한 소스

검토 대상:

- `ref/URM/README.md`
- `ref/URM/scripts/URM_sudoku.sh`
- `ref/URM/config/arch/urm.yaml`
- `ref/URM/models/urm/urm.py`
- `ref/URM/pretrain.py`

URM README의 주장:

- Universal Transformer 계열에서 성능 향상의 주요 원인은 복잡한 설계 자체보다 recurrent
  inductive bias와 Transformer의 강한 nonlinear component에 있다고 해석한다.
- URM은 short convolution과 truncated backpropagation을 추가했다고 설명한다.
- TRM paper의 Sudoku `87.4%`는 MLP model 결과라 ARC용 TRM architecture와 다르다고 보고,
  Sudoku 재현 비교에는 attention TRM architecture를 사용한다고 밝힌다.

이 README 주장은 그대로 받아들이기보다는, 아래처럼 실제 스크립트와 코드 구조 기준으로 해석해야 한다.

## 2. URM Sudoku 실행 설정

`ref/URM/scripts/URM_sudoku.sh`:

```bash
torchrun --nproc-per-node 8 pretrain.py \
  data_path=PATH_TO_SUDOKU \
  arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
  epochs=50000 \
  eval_interval=2000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  +run_name=$run_name \
  +checkpoint_path=$checkpoint_path \
  +ema=True
```

주의점:

- `torchrun --nproc-per-node 8`이지만 `global_batch_size=128`이다. 즉 8 GPU면 local batch는 대략
  GPU당 16이다.
- `epochs=50000`이고, Sudoku metadata가 `total_groups=1000`, `mean_puzzle_examples=1.0`이면
  optimizer step 수는 `390625`다.
- `eval_interval=2000` epoch이므로 eval은 대략 `2000*1000/128 = 15625` optimizer step마다 돈다.
- 현재 RVSAC 실험도 `batch=128, eval_interval=2000`이면 `step 15625, 31250, ...`에서 eval이
  찍히므로 중간 step 비교 자체는 자연스럽다.

`ref/URM/config/arch/urm.yaml`의 기본값은 `num_layers=8`, `H_cycles=4`, `L_cycles=3`이지만,
Sudoku 스크립트가 `num_layers=4`, `H_cycles=2`, `L_cycles=6`으로 덮어쓴다.

## 3. URM 구조

URM carry는 TRM처럼 H/L 두 state가 아니라 single hidden 하나다.

```text
carry:
  current_hidden
  steps
  halted
  current_data
```

`URM_Inner.forward`의 핵심은 다음 구조다.

```python
input_embeddings = concat(puzzle_emb, token_embed(input))
hidden_states = carry.current_hidden

with torch.no_grad():
    for _ in range(H_cycles - 1):
        for _ in range(L_cycles):
            hidden_states = hidden_states + input_embeddings
            hidden_states = layers(hidden_states)

for _ in range(L_cycles):
    hidden_states = hidden_states + input_embeddings
    hidden_states = layers(hidden_states)

new_carry.current_hidden = hidden_states.detach()
logits = lm_head(hidden_states)
q_logits = q_head(hidden_states[:, 0])
```

Sudoku script에서는 `H_cycles=2`, `L_cycles=6`, `num_layers=4`이므로 한 optimizer step 안에서:

```text
no_grad layer-stack passes = (H_cycles - 1) * L_cycles = 6
grad layer-stack passes    = L_cycles = 6

각 stack pass는 URMBlock 4개.

no_grad blocks = 6 * 4 = 24
grad blocks    = 6 * 4 = 24
```

이 점이 중요하다. URM은 layer가 4라서 깊어졌지만, 절반은 no-grad burn-in이고 절반만 gradient를
가진다.

URMBlock은 다음 구성이다.

```text
Attention
RMSNorm residual
ConvSwiGLU
RMSNorm residual
```

`ConvSwiGLU`는 기존 SwiGLU의 dense projection 사이에 depthwise Conv1d(kernel=2)를 넣는다.
hidden=512, expansion=4이면 intermediate는 TRM/RVSAC SwiGLU와 같은 `1536`이고, depthwise conv
비용은 dense matmul 대비 작다. FLOPs 비교에서는 block 수와 grad/no_grad 비율이 지배적이다.

## 4. TRM/URM/RVSAC 한 optimizer step 비교

공통 기준:

```text
B = 128
seq_len = 81 + puzzle_emb_len 16 = 97
hidden = 512
num_heads = 8
expansion = 4
SwiGLU intermediate = 1536
```

이 조건에서 attention+SwiGLU block 1개 forward는 이전 FLOPs 문서 기준 대략:

```text
block_fwd ~= 87.09 GFLOPs
```

URM의 depthwise conv는 대략:

```text
conv_fwd ~= 2 * B * L * inter * kernel
         ~= 2 * 128 * 97 * 1536 * 2
         ~= 0.076 GFLOPs
```

즉 block당 dense 비용의 `0.1%` 수준이라 1차 비교에서는 무시 가능하다.

### 4.1 Block-unit 비교

Backward 배수는 이전 문서와 동일하게 둔다.

```text
no_grad block = 1 unit
grad block    = 3 units   # forward + dx + dW
```

| 모델 | no_grad block | grad block | block-unit |
|---|---:|---:|---:|
| TRM attention L2, H3 L6 | 28 | 14 | `28 + 3*14 = 70` |
| URM Sudoku L4, H2 L6 | 24 | 24 | `24 + 3*24 = 96` |
| RVSAC H32 L2 full | 0 | 64 | `3*64 = 192` |

따라서 batch 128 기준 per-step dense FLOPs는 대략:

```text
TRM attention batch128  ~= 70  * 87.09 GF = 6.10 TFLOPs / step
URM Sudoku batch128     ~= 96  * 87.09 GF = 8.36 TFLOPs / step
RVSAC H32 L2 full       ~= 192 * 87.09 GF = 16.72 TFLOPs / step
```

RVSAC은 transition마다 `out_proj(512->512)`가 추가되고 value/lm head도 있으므로 기존 정밀 계산에서는
`~17.37 TFLOPs / step`으로 잡았다.

결론:

- URM은 TRM attention batch128보다 per-step이 가볍지 않다. 오히려 대략 `1.37x` 무겁다.
- URM은 RVSAC H32 full보다 per-step은 가볍다. 대략 `0.48x` 수준이다.
- URM의 효율 주장은 "항상 더 적은 FLOPs"라기보다, TRM류 구조를 더 단순하게 만들고 성능을 높였다는
  쪽으로 읽어야 한다.

## 5. 전체 학습 FLOPs 비교

### 5.1 현재 RVSAC 8000 epoch 기준

```text
RVSAC H32 L2 full:
  per_step ~= 17.37 TFLOPs
  steps    = 62500
  total    ~= 1.09 EFLOPs
```

```text
URM Sudoku script:
  per_step ~= 8.36 TFLOPs
  steps    = 390625
  total    ~= 3.27 EFLOPs
```

이 기준에서는 URM script 전체 학습 compute가 RVSAC 8000 epoch보다 약 `3.0x` 크다.

### 5.2 official TRM attention command와 비교

URM README에 제시된 attention TRM Sudoku 재현 커맨드는 `global_batch_size`를 지정하지 않는다.
TRM config 기본값이 `768`이라면:

```text
official TRM attention:
  batch    = 768
  epochs   = 50000
  steps    = 50000 * 1000 / 768 ~= 65104
  per_step ~= 6.10 TFLOPs * 6 = 36.6 TFLOPs
  total    ~= 2.38 EFLOPs
```

따라서 rough dense-FLOPs 기준:

```text
RVSAC H32 8000 epochs     ~= 1.09 EFLOPs
official TRM attention    ~= 2.38 EFLOPs
URM Sudoku script         ~= 3.27 EFLOPs
```

즉 공식 attention TRM 또는 URM 최종 결과와 비교할 때는 RVSAC 8000 epoch가 compute를 덜 쓴 상태일
수 있다. 반대로 같은 batch128, 같은 step에서 비교하면 RVSAC H32 full은 URM보다 per-step이 약 2배
무겁다.

## 6. Step 기준 비교의 의미

RVSAC과 URM을 둘 다 `batch=128`로 두면, `step 31250`에서 batch-slot exposure는 둘 다:

```text
31250 * 128 = 4,000,000 slot presentations
```

그러나 ACT/carry 모델은 non-halted slot이 새 dataloader sample을 버리고 이전 sample을 계속 사용한다.
따라서 "고유 sample episode 수"는 평균 halt length에 따라 줄어든다.

URM도 TRM과 같은 carry slot 구조를 쓴다.

```python
new_current_data = where(carry.halted, batch, carry.current_data)
```

즉 halted slot만 새 sample을 받고, 아직 안 멈춘 slot은 dataloader가 준 새 sample을 무시한다.

이 구조 때문에 단순 `optimizer step`, `batch slot`, `unique sample episode`는 서로 다르다.

```text
fixed-rollout RVSAC:
  unique presentations ~= N * B

TRM/URM carry-ACT:
  unique episodes ~= N * B / avg_halt_steps
```

다만 URM/TRM은 같은 sample을 여러 optimizer step에 걸쳐 업데이트하므로, sample을 덜 본다고 해서
학습 신호가 약하다는 뜻은 아니다. 오히려 sample당 여러 local reasoning updates를 하는 구조다.

## 7. RVSAC에 주는 시사점

URM에서 RVSAC에 바로 가져올 만한 후보는 다섯 가지다.

### 7.1 Input injection

URM은 매 loop마다:

```text
hidden_states = hidden_states + input_embeddings
hidden_states = layers(hidden_states)
```

TRM도 같은 계열의 input injection을 쓴다. RVSAC은 `h0`에만 input이 들어가고 이후에는
`h_{t+1}=h_t+f(h_t)`로 흐른다. outer residual 때문에 입력 정보가 완전히 사라지지는 않지만, 긴 rollout에서
transition이 자체 attractor를 만들면 input-conditioned signal이 약해질 수 있다.

RVSAC에 넣는다면 안전한 형태는 state 자체에 input을 계속 누적하는 방식이 아니라 transition 입력에만 넣는
방식이다.

```text
u_t = f_theta(rms_norm(h_t + alpha * e))
h_{t+1} = h_t + u_t
```

피해야 할 형태:

```text
h_{t+1} = h_t + alpha * e + f_theta(h_t)
```

이 방식은 input이 horizon만큼 누적되어 state scale과 gradient를 키울 수 있다.

### 7.2 Carry-ACT sample reuse

URM은 TRM처럼 carry slot을 유지한다. RVSAC H32 full은 매 optimizer step마다 새 batch 전체를 32-step
rollout한다.

공정 비교용 RVSAC ablation은 다음처럼 둘 수 있다.

```text
RVSAC Carry-ACT K=8:
  한 optimizer step에서는 8-step segment만 rollout
  non-halted slot은 같은 sample/current hidden 유지
  halted slot만 새 dataloader sample 수용
  max_segments=4이면 최대 depth=32
```

이 구조는 현재 `arch.horizon=32, arch.bptt_segment=8`과 다르다. 현재 K=8은 한 optimizer step 안에서
4개 segment를 모두 처리한다. Carry-ACT K=8은 한 step에 segment 하나만 처리하고 carry를 다음 optimizer
step으로 넘긴다.

### 7.3 No-grad burn-in

URM은 한 optimizer step 안에서 절반을 no-grad로 돌리고, 마지막 절반만 gradient를 태운다.

RVSAC에도 다음 ablation을 둘 수 있다.

```text
H = 32
burnin = 16 or 24

for t < burnin:
  with torch.no_grad():
    h = h + f(h)

for t >= burnin:
  h = h + f(h)       # grad
```

이 경우 per-step FLOPs는 크게 줄고, gradient chain은 짧아진다. 다만 RVSAC의 actor objective가
전체 reward sum을 BPTT하는 구조라는 점이 약해지므로, 목적함수 해석은 `late-stage local improvement`
쪽으로 바뀐다.

### 7.4 L_layers=4

URM Sudoku는 `num_layers=4`다. RVSAC에서 단순히 `arch.L_layers=4`를 켜면 block 수가 두 배가 되고,
H32 full-BPTT에서는 per-step FLOPs도 거의 두 배가 된다.

따라서 `L_layers=4` 실험은 다음 중 하나와 같이 묶는 편이 낫다.

```text
H=16, L_layers=4
H=24, L_layers=4
H=32, L_layers=4, K=8
H=32, L_layers=4, no_grad burn-in
Carry-ACT K=8, L_layers=4
```

그냥 `H=32, L_layers=4, full BPTT`는 계산량이 너무 커서 URM/TRM과의 공정 비교에서 불리하다.

### 7.5 ConvSwiGLU

URM은 SwiGLU 내부의 gated activation 뒤에 depthwise Conv1d(kernel=2)를 넣는다.

```text
gate, up = gate_up_proj(x).chunk(2)
x_ffn = silu(gate) * up
x_conv = depthwise_conv1d(x_ffn)
out = down_proj(silu(x_conv))
```

FLOPs는 거의 늘지 않지만, sequence-local mixing과 nonlinear filtering이 추가된다. Sudoku에서는 행/열/박스
구조가 token ordering과 완전히 일치하지는 않으므로 효과가 보장되지는 않는다. 다만 URM의 "strong nonlinear
components" 주장과 맞닿아 있으므로, input injection과 carry-ACT 이후의 후보로 둘 만하다.

## 8. 현재 결론

URM은 "TRM보다 무조건 FLOPs가 적은 모델"로 보기 어렵다. Sudoku script만 보면:

```text
per-step:
  TRM batch128 < URM batch128 < RVSAC H32 batch128

full script:
  RVSAC 8000 epochs < official TRM attention 50000 epochs < URM 50000 epochs
```

하지만 URM이 주는 구조적 힌트는 강하다.

```text
1. 매 loop input injection
2. single hidden carry
3. carry-ACT sample reuse
4. no-grad burn-in + truncated BPTT
5. ConvSwiGLU 또는 더 강한 local nonlinear mixing
```

현재 RVSAC이 TRM/URM류와 경쟁하려면 단순히 H를 키우는 것보다, 위 다섯 요소 중 최소한
`input injection`과 `carry-ACT K-step`을 ablation으로 넣고 비교하는 것이 우선이다.

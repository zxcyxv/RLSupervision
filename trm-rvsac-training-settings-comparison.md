# TRM 학습 설정 디테일 및 RVSAC 대비 차이

## 0. 핵심 요약

TRM은 겉으로는 `halt_max_steps=16`, `H_cycles=3`, `L_cycles=6`이라 매우 긴 반복 추론 모델처럼 보이지만, 학습 역전파는 강하게 잘려 있다.

```text
TRM 한 ACT step 내부:
  total reasoning module calls = H_cycles * (L_cycles + 1) = 3 * 7 = 21
  no_grad calls                = (H_cycles - 1) * 7 = 14
  grad-tracked calls           = 7

TRM ACT step 사이:
  z_H, z_L carry는 detach
```

즉 TRM은 **긴 forward 반복 + 짧은 BPTT + carry-based online ACT** 구조다.

반면 RVSAC은 한 optimizer step 안에서 `H` step rollout을 전부 만들고, 기본 `bptt_segment=0`이면 그 전체가 actor BPTT 경로다.

```text
RVSAC H=16:
  grad-tracked transition calls = 16

RVSAC H=32 full:
  grad-tracked transition calls = 32

RVSAC H=32,K=8:
  forward transition calls      = 32
  max gradient chain            = 8
```

따라서 TRM과 RVSAC은 같은 hidden size, block, data pipeline을 써도 학습 역학이 상당히 다르다. TRM의 반복횟수 증가는 대부분 forward/carry 쪽이고, RVSAC의 반복횟수 증가는 full BPTT일 때 gradient chain 증가다.

## 1. TRM 공식/참조 Sudoku 학습 레시피

`ref/TinyRecursiveModels/README.md`의 Sudoku-Extreme 레시피:

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=pretrain_att_sudoku ema=True
```

MLP-over-sequence variant:

```bash
arch.mlp_t=True arch.pos_encodings=none
```

README상 기대 성능:

```text
Sudoku-Extreme MLP-t:  ~87% exact accuracy
Sudoku-Extreme attn:   ~75% exact accuracy
```

주의할 점:

- README Sudoku 명령은 `global_batch_size`를 따로 지정하지 않으므로 기본값 `768`을 사용한다.
- 1 GPU batch를 낮춰 돌리면 총 optimizer step 수가 늘어난다.
- `epochs=50000`, `global_batch_size=768`, Sudoku 1000 groups이면 대략 `50000*1000/768 ~= 65k` optimizer steps다.
- 우리가 쓴 `epochs=8000`, `global_batch_size=128`도 `8000*1000/128 = 62.5k`라 step 수 기준으로는 비슷하다.

## 2. TRM 기본 config

`ref/TinyRecursiveModels/config/arch/trm.yaml`:

| 항목 | 값 |
|---|---:|
| `hidden_size` | 512 |
| `num_heads` | 8 |
| `expansion` | 4 |
| `L_layers` | 2 |
| `H_cycles` | 3 |
| `L_cycles` | 6 |
| `halt_max_steps` | 16 |
| `halt_exploration_prob` | 0.1 |
| `puzzle_emb_ndim` | hidden size |
| `puzzle_emb_len` | 16 |
| `pos_encodings` | rope |
| `forward_dtype` | bfloat16 |
| `loss_type` | stablemax_cross_entropy |
| `no_ACT_continue` | True |

`H_layers`는 config에 있지만 TRM 코드에서는 ignored다. 실제 reasoning module은 `L_level` 하나를 공유해서 `z_L`, `z_H` 업데이트에 반복 사용한다.

## 3. TRM 재귀 구조

TRM은 두 latent state를 쓴다.

```text
z_H: high-level latent
z_L: low-level latent
```

초기값은 학습 parameter가 아니라 buffer다.

```python
H_init = Buffer(trunc_normal(...))
L_init = Buffer(trunc_normal(...))
```

halted slot은 다음 sample로 바뀔 때 `H_init`, `L_init`로 reset된다.

### 3.1 입력 재주입

TRM reasoning module은 매 호출마다 input injection을 더한다.

```python
hidden_states = hidden_states + input_injection
for block in layers:
    hidden_states = block(hidden_states)
```

내부 반복은 다음 꼴이다.

```text
for H_step in H_cycles - 1:
  for L_step in L_cycles:
    z_L = L_level(z_L, z_H + input_embeddings)  # no_grad
  z_H = L_level(z_H, z_L)                       # no_grad

for L_step in L_cycles:
  z_L = L_level(z_L, z_H + input_embeddings)    # grad
z_H = L_level(z_H, z_L)                         # grad
```

기본값에서는:

```text
no_grad: 2 * (6 + 1) = 14 module calls
grad:    1 * (6 + 1) = 7 module calls
```

즉 TRM은 **forward로 state를 충분히 예열한 뒤 마지막 cycle만 gradient**를 태운다.

### 3.2 carry detach

TRM inner forward 끝:

```python
new_carry = (z_H.detach(), z_L.detach())
```

따라서 ACT step 사이로 BPTT가 이어지지 않는다.

이 점이 RVSAC full BPTT와 가장 큰 차이다. TRM이 16 ACT step까지 반복하더라도, 각 optimizer step에서 gradient chain은 7 module calls 근방으로 제한된다.

## 4. TRM ACT slot 시스템

TRM `train_state.carry`는 optimizer step 사이에 유지된다.

매 train batch에서:

```python
new_inner_carry = reset_carry(carry.halted, carry.inner_carry)
new_current_data = where(carry.halted, batch, carry.current_data)
```

즉 batch의 각 slot은:

- 이전 puzzle이 halted면 새 batch sample로 교체
- 아직 halted가 아니면 이전 puzzle을 계속 진행

학습 loop는 매 optimizer step마다 ACT 한 step만 진행한다.

```python
carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
loss.backward()
optimizer.step()
```

이 구조 때문에 TRM의 `epochs`와 `batch` 해석은 일반 supervised loop와 다르다. DataLoader는 매 step 새 batch를 제공하지만, active slot은 그 batch를 무시하고 이전 puzzle을 계속 푼다.

## 5. TRM halting 설정

기본:

```text
halt_max_steps = 16
halt_exploration_prob = 0.1
no_ACT_continue = True
```

training 중에는:

```python
halted = is_last_step
if training and halt_max_steps > 1:
    halted = halted | (q_halt_logits > 0)
    min_halt_steps = Bernoulli(0.1) * randint(2, halt_max_steps+1)
    halted = halted & (steps >= min_halt_steps)
```

의미:

- `q_halt_logits > 0`이면 조기 halt 가능
- 하지만 exploration 때문에 일부 sample은 랜덤 최소 step까지 강제로 계속 간다
- `is_last_step`이면 무조건 halt

eval 중에는 `self.training == False`이므로 조기 halt 로직이 꺼진다.

```text
eval에서는 항상 max step까지 돈다.
```

그래서 TRM의 q-head는 eval compute를 줄이는 halting policy라기보다:

- training 중 ACT slot 교체 시점 결정
- `q_halt_loss`의 보조 신호
- ARC voting/ranking score

역할에 가깝다.

## 6. TRM loss

TRM loss:

```text
L = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
```

기본 `no_ACT_continue=True`라서 `q_continue_loss = 0`이다.

따라서 실제 기본 손실은:

```text
L = stablemax CE + 0.5 * BCEWithLogits(q_halt_logits, seq_is_correct)
```

여기서 `seq_is_correct`는 현재 step의 전체 sequence exact correctness다.

특징:

- 모든 ACT step에서 LM loss가 걸린다.
- q_halt는 현재 출력이 exact인지 맞추도록 학습된다.
- Bellman/critic/value target은 없다.
- q_continue Q-learning 계열 코드는 남아 있지만 기본 설정에서는 꺼져 있다.

TRM metrics:

```text
accuracy
exact_accuracy
q_halt_accuracy
steps
lm_loss
q_halt_loss
```

그리고 metric 집계는 `valid_metrics = new_carry.halted & valid`만 대상으로 한다. 즉 train log의 accuracy/exact는 halted slot만 반영한다.

## 7. TRM optimizer와 LR 설정

TRM pretrain은 원본 기준 `adam_atan2.AdamATan2`를 필수로 import한다.

```python
from adam_atan2 import AdamATan2
```

우리 RVSAC pretrain은 import 실패 시 AdamW fallback을 둔다.

```python
AdamATan2 if available else AdamW
```

TRM optimizer 분리:

```text
1. puzzle embedding: CastedSparseEmbeddingSignSGD_Distributed
2. 나머지 model params: AdamATan2
```

Sudoku README 레시피:

```text
lr = 1e-4
puzzle_emb_lr = 1e-4
weight_decay = 1.0
puzzle_emb_weight_decay = 1.0
beta1 = 0.9
beta2 = 0.95
```

기본 config는 ARC용에 가깝다.

```text
global_batch_size = 768
epochs = 100000
eval_interval = 10000
weight_decay = 0.1
puzzle_emb_weight_decay = 0.1
puzzle_emb_lr = 1e-2
ema = False
```

Sudoku에서는 README 명령으로 `puzzle_emb_lr`, weight decay, EMA 등을 override한다.

### 7.1 LR schedule의 특이점

코드는 cosine schedule 함수지만, 기본 `lr_min_ratio=1.0`이면 warmup 이후 decay가 없다.

```python
lr = base_lr * (min_ratio + ...)
```

`min_ratio=1.0`이면 항상 base lr이다.

즉 실질적으로:

```text
first 2000 steps: linear warmup
after warmup: constant lr
```

이다.

## 8. TRM eval/checkpoint 동작

TRM eval은 train과 다르게 각 eval batch마다 carry를 새로 만든다.

```python
carry = model.initial_carry(batch)
while True:
    carry, loss, metrics, preds, all_finish = model(...)
    if all_finish:
        break
```

eval에서는 조기 halt가 꺼져 있으므로 기본적으로 `halt_max_steps=16`까지 돈다.

원본 TRM checkpoint는 eval 이후 저장된다.

```text
train segment -> eval -> save checkpoint
```

따라서 eval이 죽으면 그 eval interval의 checkpoint가 남지 않는다. 우리 RVSAC은 이 문제가 실제로 발생했기 때문에 지금은 eval 전에 raw checkpoint를 먼저 저장하도록 바꿨다.

## 9. RVSAC 현재 설정 요약

RVSAC 기본 config:

| 항목 | 값 |
|---|---:|
| `hidden_size` | 512 |
| `num_heads` | 8 |
| `expansion` | 4 |
| `L_layers` | 2 |
| `horizon` | 16 기본, 실험으로 32 |
| `bptt_segment` | 0 기본, 실험으로 8 |
| `gamma` | 0.95 |
| `lam` | 0.9 |
| `beta` | 0.1 |
| `critic_coef` | 1.0 |
| `target_tau` | 0.005 |
| `critic_shapes_trunk` | False |
| `loss_type` | stablemax_cross_entropy |

h16 실험 설정:

```text
global_batch_size = 128
epochs = 8000
eval_interval = 2000
lr = 1e-4
puzzle_emb_lr = 1e-4
weight_decay = 1.0
puzzle_emb_weight_decay = 1.0
ema = True
horizon = 16
```

이 설정은 총 step 수 기준으로 TRM README Sudoku 기본 batch 768, epochs 50000과 비슷하다.

```text
RVSAC: 8000 * 1000 / 128 = 62,500 steps
TRM README default batch: 50000 * 1000 / 768 ~= 65,104 steps
```

## 10. TRM vs RVSAC 차이표

| 항목 | TRM | RVSAC |
|---|---|---|
| 학습 단위 | ACT 한 step | H-step rollout 한 번 |
| long inference | carry를 여러 optimizer step에 걸쳐 진행 | 한 forward 안에서 H step rollout |
| state | `z_H`, `z_L`, `current_data`, `steps`, `halted` carry | `h_0..h_H` local rollout |
| input injection | 매 L update마다 `z_H + input_embeddings` 재주입 | 입력은 `h_0`에만 embedding, 이후 재주입 없음 |
| recurrence | `L_level` 공유 모듈로 z_L/z_H 업데이트 | `h <- h + f(h)` residual transition |
| no_grad burn-in | 있음, 기본 14 module calls | 없음 |
| grad depth | 기본 7 module calls/optimizer step | full: H, segmented: K |
| carry detach | 매 ACT step 후 detach | full BPTT는 rollout 내부 detach 없음 |
| loss | CE + halt BCE | TD(lambda) critic + actor return |
| critic/value | 없음. q_halt는 halt/exact predictor | scalar value head + target value head |
| halting | training에서 q_halt + exploration, eval에서는 max step | 없음. H 고정 |
| metric 집계 | halted slot 중심 | 전체 batch final h_H |
| optimizer | AdamATan2 필수 + sparse SignSGD | AdamATan2 fallback AdamW + sparse SignSGD |
| EMA | option, Sudoku README에서 True | option, h16/h32 실험에서 True |
| LR decay | warmup 후 사실상 constant | 동일 schedule 계승 |
| eval | max ACT steps loop | single H-step rollout |

## 11. 비교 시 주의해야 할 함정

### 11.1 "같은 step"이 같은 의미가 아니다

TRM의 1 optimizer step은 ACT slot을 한 번 진전시키는 것이다. 일부 slot은 같은 puzzle을 이어서 풀고, halted slot만 새 puzzle로 교체된다.

RVSAC의 1 optimizer step은 batch 전체 puzzle에 대해 H-step rollout을 완성한다.

따라서 train log의 `step N`을 단순 비교하면 안 된다. 같은 optimizer step 수라도 sample exposure, repeated intermediate supervision, carry continuation 구조가 다르다.

### 11.2 TRM train accuracy는 halted slot 기반이다

TRM train metric은 `new_carry.halted`인 slot만 집계한다. 조기 halt가 쉬운 sample이 먼저 metric에 잡힐 수 있다.

RVSAC train metric은 batch 전체의 final state 기준이다.

즉 TRM train accuracy와 RVSAC train accuracy의 통계적 의미가 다르다. test/eval exact accuracy가 더 공정하다.

### 11.3 TRM은 BPTT 안정성을 구조적으로 산다

TRM은 `no_grad` burn-in과 carry detach 때문에 긴 반복을 써도 gradient chain이 짧다. RVSAC full H=32는 실제로 32-step BPTT라 안정성 조건이 더 빡세다.

H=32,K=8 RVSAC은 TRM 쪽 truncation 철학에 더 가깝지만, 현재 로그상 K=8에서는 critic lag와 작은 per-step update 때문에 H 증가 이점이 잘 안 보였다.

### 11.4 TRM은 input을 계속 재주입한다

TRM은 매 L update에서 `z_H + input_embeddings`를 넣는다. 이는 장기 반복 중 입력 정보가 희석되는 것을 막는다.

RVSAC은 입력이 `h_0`에만 들어가고 이후 순수 residual transition이다. H를 늘릴 때 입력 정보 보존/재주입 여부는 중요한 ablation 후보다.

## 12. RVSAC에서 TRM식 디테일을 차용한다면

우선순위가 높은 후보:

1. **eval-safe checkpointing**
   이미 적용. eval 전에 raw checkpoint 저장.

2. **warmup 후 constant LR**
   이미 동일. `lr_min_ratio=1.0`.

3. **MLP-t Sudoku variant**
   TRM README상 Sudoku에서는 `mlp_t=True`, `pos_encodings=none`이 attention보다 높게 나온다. RVSAC에서도 `arch.mlp_t=True arch.pos_encodings=none` ablation 가치가 크다.

4. **input reinjection**
   TRM은 반복마다 입력을 재주입한다. RVSAC pure residual이 plateau를 보이면 `f(h, input_embedding)` 또는 약한 input skip/gate를 실험할 가치가 있다.

5. **truncated BPTT with longer forward**
   TRM식으로 보면 `H=32,K=8`은 방향은 맞다. 다만 현재 K=8 로그는 H=16보다 좋지 않았다. `K=16`, `H=32` 또는 `H=48,K=16`이 더 적절할 수 있다.

6. **halt/exact auxiliary head**
   RVSAC에는 halt head가 없고 terminal value만 있다. exact correctness를 예측하는 보조 head를 추가하면 ARC voting이나 adaptive compute에 도움이 될 수 있지만, 현재 RVSAC 이론의 value head와 충돌하지 않도록 별도 auxiliary로 둬야 한다.

## 13. 현재 실험 해석에 직접 연결되는 결론

H=32,K=8 로그에서 H=16보다 `ce_first - ce_last` gap이 커지지 않은 이유는 TRM 관점에서 보면 자연스럽다.

TRM은 길게 돌면서도:

```text
1. input reinjection
2. no_grad burn-in
3. carry continuation
4. short gradient depth
5. halt/exact auxiliary
```

를 동시에 쓴다.

반면 RVSAC H=32,K=8은:

```text
1. input reinjection 없음
2. K=8마다 gradient 단절
3. boundary 이후 미래는 value bootstrap에 의존
4. critic_loss/td_abs가 H16보다 큼
```

상태였다. 따라서 단순히 H를 32로 늘리고 K=8로 자르는 것만으로 TRM식 장기 반복 효과가 바로 나오지 않는 것은 이상하지 않다.

다음 비교에서 가장 공정한 축은 세 개다.

```text
A. RVSAC H=16 full
B. RVSAC H=32 full
C. RVSAC H=32,K=16
```

그리고 TRM과 직접 맞추려면 별도 ablation:

```text
D. RVSAC H=32 full + input reinjection
E. RVSAC H=32,K=16 + input reinjection
F. RVSAC mlp_t=True,pos_encodings=none
```

를 보는 것이 좋다.

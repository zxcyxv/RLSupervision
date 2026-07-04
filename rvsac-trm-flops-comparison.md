# RVSAC(H=32,L=2) vs TRM — 1 optimizer step / 전체 훈련 FLOPs 정밀 비교

기준 커맨드 (사용자 제공, 현재 실행 중인 TRM 런):

```
global_batch_size=128 arch.hidden_size=512 arch.num_heads=8 arch.L_layers=2
arch.H_cycles=3 arch.L_cycles=6  (arch/trm.yaml 기본값: expansion=4, puzzle_emb_ndim=512, puzzle_emb_len=16, halt_max_steps=16, no_ACT_continue=True)
epochs=8000 eval_interval=2000, data=sudoku-extreme-1k-aug-1000 (seq_len=81, vocab=11)
```

RVSAC 비교 대상: `arch.horizon=32 arch.L_layers=2`, 나머지(hidden_size=512, num_heads=8, expansion=4,
puzzle_emb_ndim=512, puzzle_emb_len=16, bptt_segment=0 기본값=풀 BPTT), 동일 batch/dataset.

## 0. 결론 요약

| | 1 optimizer step | 전체 훈련 (62,500 step) |
|---|---|---|
| **TRM** (H_cycles=3, L_cycles=6, L=2) | **6.097 TFLOPs** | **381.0 PFLOPs** |
| **RVSAC** (H=32, L=2, 풀 BPTT) | **17.367 TFLOPs** | **1085.4 PFLOPs** |
| **비율** | **≈2.85배** | **≈2.85배 (동일)** |

- **Q1 (1 step 차이)**: RVSAC(H=32,L=2)가 TRM보다 **약 2.85배** 많은 FLOPs를 씁니다.
- **Q2 (전체 훈련 시 동일한가)**: **optimizer step 횟수는 정확히 동일(62,500)합니다** — `total_steps = epochs × total_groups × mean_puzzle_examples / global_batch_size`는 데이터셋·에폭·배치크기만의 함수이고 모델 구조와 무관하기 때문입니다(양쪽 `pretrain.py`가 동일 공식 사용, 코드 확인). 따라서 **전체 훈련 FLOPs 비율도 1-step 비율과 정확히 같은 ≈2.85배**입니다 — step 수가 상쇄되기 때문에 당연한 결과입니다.
- 실측 검증: 컴파일된 실제 런에서 TRM 149.25ms/step(6.70it/s, 지금 도는 프로세스), RVSAC(H=16, 이전 런) 210ms/step(4.76it/s) → 실측 비율 1.407, 아래서 계산할 해석적(analytic) 비율(H=16 기준) 1.424와 **오차 <2%** — FLOPs 기반 추정이 실측과 잘 맞습니다.

---

## 1. 공통 블록: Attention+SwiGLU (양쪽이 완전히 동일한 레이어)

TRM의 `TinyRecursiveReasoningModel_ACTV1Block`과 RVSAC의 `RVSACBlock`은 **동일한 클래스**(`rvsac/layers.py`가 TRM `models/layers.py`를 그대로 복사한 것)를 씁니다. hidden=512, num_heads=8, head_dim=64, expansion=4 → SwiGLU 내부 `inter = _find_multiple(round(4·512·2/3), 256) = 1536`(양쪽 동일 계산).

시퀀스 길이 `L = seq_len + puzzle_emb_len = 81 + 16 = 97` (양쪽 동일 — RVSAC도 TRM과 같은 puzzle embedding 접두 방식 사용). `B = 128`.

FLOP 컨벤션: 표준 `2×곱셈덧셈` 방식(Kaplan/Hoffmann 스케일링 로우 컨벤션)만 세고, RMSNorm/RoPE/softmax/임베딩 lookup 등 elementwise/O(L) 연산은 주요 matmul 대비 <1%라 생략합니다(각주에 근거 수치 명시).

### 1.1 Attention 블록 (forward, B=128, L=97)

- `qkv_proj`: Linear(512 → 24·64=1536): `2·(B·L)·512·1536 = 2·12416·786432 = 1.95287×10¹⁰`
- attention core (QKᵀ + softmax·V, 헤드 합산): `4·B·L²·512 = 4·128·9409·512 = 2.46651×10⁹`
- `o_proj`: Linear(512→512): `2·12416·512·512 = 6.50956×10⁹`
- **Attention 합계 (forward) = 2.85048×10¹⁰**

### 1.2 SwiGLU MLP 블록 (forward)

- `gate_up_proj`(512→3072) + `down_proj`(1536→512): `2·B·L·512·(3072+1536) = 2·12416·512·4608 = 5.85860×10¹⁰`

### 1.3 블록 1개 forward 합계

```
FLOPs_block(fwd) = 2.85048e10 + 5.85860e10 = 8.70908×10¹⁰   (B=128, L=97 전체 기준)
```

### 1.4 역전파 배수 규칙 (핵심 — 여기서 두 모델이 갈립니다)

행렬곱 `y = xW`에서 backward는 `dx = dy·Wᵀ`, `dW = xᵀ·dy` 두 항으로, 각각 forward와 같은 비용. PyTorch autograd는 **입력이 `requires_grad=False`면 `dx` 계산을 건너뛰고, 파라미터가 `requires_grad=False`(freeze)면 `dW` 계산을 건너뜁니다**(`needs_input_grad` 플래그, 실제 동작). 따라서:

| 상황 | forward | backward | 합계 배수 |
|---|---|---|---|
| 학습 대상 파라미터 + 그래프에 연결된 입력 (일반적인 경우) | 1× | dx+dW = 2× | **3×** |
| 입력이 detach됨 (dx 불필요), 파라미터는 학습 | 1× | dW만 = 1× | **2×** |
| 파라미터 freeze(dW 불필요), 입력은 그래프에 연결(dx 필요) | 1× | dx만 = 1× | **2×** |
| `torch.no_grad()` 블록 내부 | 1× | 0 | **1×** |

---

## 2. TRM 1 optimizer step 연산 그래프

`TinyRecursiveReasoningModel_ACTV1_Inner.forward` (`trm.py:196`):

```python
with torch.no_grad():
    for _H_step in range(H_cycles-1):        # 2회
        for _L_step in range(L_cycles):      # 6회
            z_L = L_level(z_L, z_H+input)    # no_grad
        z_H = L_level(z_H, z_L)              # no_grad
for _L_step in range(L_cycles):              # 6회
    z_L = L_level(z_L, z_H+input)            # grad
z_H = L_level(z_H, z_L)                      # grad
```

`L_level` 1회 호출 = `L_layers=2`개 블록. 총 `L_level` 호출 수 = `H_cycles·(L_cycles+1) = 3·7 = 21`회:
- **no_grad**: `(H_cycles-1)·(L_cycles+1) = 2·7 = 14`회 → 28개 블록, **1× 배수**
- **grad**: `L_cycles+1 = 7`회 → 14개 블록, **3× 배수**

이후 `no_ACT_continue=True`(기본값, 커맨드에서 변경 안 함)이므로 Q-continue용 2차 forward는 **건너뜀**(`trm.py:289` 분기 미실행). `lm_head`는 `z_H`(97토큰) 전체에 적용 후 `[:, 16:]`로 슬라이스(81토큰만 쓰지만 **97토큰만큼 연산**함 — 구현상 낭비이나 RVSAC도 동일 패턴), `q_head`는 `z_H[:,0]` 1토큰.

```
no_grad 항 = 14 × FLOPs_block(fwd) = 14 × 8.70908e10 = 2.438542×10¹²
grad 항   = 7  × FLOPs_block(fwd) × 3 = 7 × 2.612724e11 = 3.657813×10¹²
lm_head   = 2·B·97·512·11 × 3(grad) = 1.398538e8 × 3 = 4.195614×10⁸
q_head    = 2·B·1·512·2 × 3(grad)  = 2.62144e5 × 3   = 7.86432×10⁵   (무시 가능)

TRM 1-step 합계 = 2.438542e12 + 3.657813e12 + 4.195614e8 + 7.86432e5
              ≈ 6.09678×10¹² FLOPs  (≈ 6.097 TFLOPs)
```

---

## 3. RVSAC(H=32, L=2) 1 optimizer step 연산 그래프

`RVSAC.forward` (`rvsac/model.py:192`), `bptt_segment=0` → 매 스텝 `h`를 detach하지 않음 → **h_0→h_32 전체가 하나의 BPTT 체인**.

루프 매 스텝(t=0..31):
```python
u = self.transition(h, cos_sin)   # 2 blocks + out_proj(512→512), 전부 grad(학습중이고 체인에 연결)
h = h + u
```

이후 각 도착상태 `h_1..h_32`에:
- `lm_head` (grad, 97토큰): 보상 계산용, actor loss로 이어짐
- `value_head`(학습 대상) on **detached** `h_0..h_31`: `critic_states = [s.detach() for s in states[:-1]]` (기본, `critic_shapes_trunk=False`)
- `value_head_target`(frozen) on `h_1..h_32`: **`torch.no_grad()`** 블록 안(target_values)
- `value_head_target(h_32)` 1회: `terminal_value` — **no_grad 밖**, 파라미터 freeze지만 `h_32`가 체인에 연결돼 있어 dx만 필요
- `value_head_target(h_32)` **한 번 더**: `segment_terminal_values`(비-segment 모드에서 `segment_ends=[H]`이라 `terminal_value`와 완전히 동일한 입력을 또 계산 — **구현상 중복 호출**, 아래 §5에서 지적)

### 3.1 항목별 계산 (B=128, L=97, hidden=512)

**(A) TransitionHead** = 2블록 + out_proj(512→512, TRM `L_level`엔 없는 항):
```
FLOPs(fwd) = 2×8.70908e10 + 6.50956e9 = 1.806912×10¹¹   (per step)
32 steps, 전부 3× (dx: 체인 유지 위해 h_0까지 역전파 필요, dW: transition 학습)
= 32 × 1.806912e11 × 3 = 1.734636×10¹³
```

**(B) lm_head** (97토큰, TRM과 동일 패턴):
```
1.398538e8 × 3 × 32 = 1.342596×10¹⁰
```

**(C) value_head (학습 대상, detached 입력)** — fc1(512→512)은 dW만(2×), fc2(512→1)는 dx+dW(3×, fc2→fc1 체인 유지 위해):
```
per call ≈ 2×6.71089e7(fc1) + 3×1.31072e5(fc2) = 1.346110e8
× 32 = 4.307552×10⁹
```

**(D) value_head_target for target_values** — 완전히 `no_grad()` 내부, 1×:
```
per call ≈ 6.72400e7 × 32 = 2.151680×10⁹
```

**(E) terminal_value** — frozen 가중치(dW 불필요) + 입력이 체인에 연결(dx 필요) → fc1,fc2 둘 다 2×:
```
2×6.71089e7 + 2×1.31072e5 = 1.344799×10⁸   (1회)
```

**(F) segment_terminal_values** (비-segment 모드에서 E와 완전 동일 재계산, §5 참조): **1.344799×10⁸ (1회, 중복)**

### 3.2 합계

```
RVSAC(H=32,L=2) 1-step = 1.734636e13 + 1.342596e10 + 4.307552e9 + 2.151680e9 + 1.344799e8 + 1.344799e8
                       ≈ 1.736651×10¹³ FLOPs  (≈ 17.367 TFLOPs)
```

**A항(TransitionHead)이 전체의 99.9%**를 차지 — B~F는 사실상 반올림 오차 수준.

---

## 4. 왜 2.85배인가 — 구조적 원인

TRM의 21회 재귀 호출 중 **14회는 순전파만(1×), 7회만 역전파 포함(3×)** → "유효 grad-호출 수" = 14×1 + 7×3 = **35 (L_level-call 단위)**.
RVSAC의 32회 TransitionHead 호출은 **전부 3×** (풀 BPTT라 예외가 없음) → 유효 grad-호출 수 = 32×3 = **96 (call 단위)**.

TRM은 "3주기 중 2주기(H_cycles-1)를 grad 없이 미리 생각"하는 구조 덕분에 반복 횟수(21회)에 비해 실제 학습 비용은 1/3(7/21)만 지불합니다. RVSAC는 매 재귀 스텝이 actor objective의 BPTT 경로 위에 있어 이런 "공짜 사고 단계"가 없고, 32스텝 전부가 최대 비용(3×)입니다. 여기에 `out_proj`(TRM `L_level`엔 없음)가 스텝당 +3.7%를 더합니다.

`bench_compare.py`의 실측(§0 요약)과 대조: H=16일 때 해석적 비율 1.424, 실측 컴파일 비율 1.407 — **오차 1.2%**, 이 FLOP 모델이 실제 하드웨어 타이밍을 잘 예측함을 확인했습니다.

---

## 5. 전체 훈련 비교 (Q2)

```
total_steps = epochs × total_groups × mean_puzzle_examples / global_batch_size
            = 8000 × 1000 × 1.0 / 128 = 62,500
```

이 공식은 `pretrain.py`(TRM/RVSAC 양쪽 모두) `init_train_state`에서 **모델 클래스와 무관하게 데이터셋 메타데이터+epochs+batch_size만으로 결정**됩니다(코드 확인, 완전히 동일한 한 줄). 따라서:

- **optimizer step 횟수: 62,500회로 정확히 동일**
- **총 FLOPs**: `step수 × step당 FLOPs`이므로 step수가 상쇄되어 **총합 비율도 1-step 비율과 완전히 같은 2.85배**

| | 총 FLOPs | 실측/예측 소요시간(컴파일) |
|---|---|---|
| TRM | 62,500 × 6.097e12 ≈ **381.0 PFLOPs** | 실측 149.25ms/step → **≈2.59시간** |
| RVSAC(H=32,L=2) | 62,500 × 1.7367e13 ≈ **1085.4 PFLOPs** | H=16 실측(210ms)에서 선형 외삽 → **≈425ms/step → ≈7.4시간** |

주의(해석상 함정, 단순 배율에 포함 안 됨): TRM은 ACT 방식이라 "1 optimizer step"이 **퍼즐 1개당 1 reasoning segment**일 뿐, `halt_max_steps=16`까지 여러 step에 걸쳐 같은 퍼즐을 반복 추론할 수 있습니다(퍼즐마다 실제 소요 segment 수는 학습된 halting에 따라 가변). RVSAC는 매 optimizer step마다 모든 샘플이 고정적으로 H=32 스텝 전부를 돕니다. 즉 "같은 step 수 = 같은 총 FLOPs 비율(2.85배)"는 산수로는 정확하지만, "같은 품질에 도달하는 데 필요한 compute"까지 같다는 뜻은 아닙니다 — TRM은 평균적으로 퍼즐당 halt_max_steps보다 적은 segment로 끝낼 수 있어 유효 추론 깊이가 데이터 의존적으로 줄어들 수 있는 반면, RVSAC는 항상 고정 깊이 32를 씁니다.

---

## 6. 구현상 발견한 사소한 비효율 (참고용, 결론에 영향 없음)

1. `rvsac/model.py:225,228` — `bptt_segment=0`(비-segment) 모드에서 `terminal_value`와 `segment_terminal_values`가 **동일한 `value_head_target(states[H])`를 두 번 계산**합니다. 전체 대비 0.0008%라 성능엔 무관하지만, `segment_terminal_values`가 `[terminal_value]`를 재사용하도록 한 줄 고치면 제거 가능합니다.
2. `lm_head`가 TRM·RVSAC 양쪽 다 puzzle-embedding 슬롯 포함 97토큰 전체에 대해 계산한 뒤 16개를 버립니다(81토큰만 필요). 두 모델이 동일한 패턴이라 상대 비교엔 영향 없지만, 절대 FLOPs 기준으로는 각각 약 +19.8%의 lm_head 낭비입니다(그래도 전체의 <0.1%).

## 7. 사용된 가정 (재현용)

- FLOP = 2×(곱셈+덧셈), RMSNorm/RoPE/softmax/embedding lookup/elementwise 연산 제외(주요 matmul 대비 <1%, 표준 스케일링-로우 컨벤션)
- Backward 배수는 §1.4 규칙 그대로(PyTorch autograd `needs_input_grad` 동작 기준)
- optimizer.step() 자체(Adam 등)의 파라미터당 상수 연산은 전체 대비 무시 가능한 수준(수백만 파라미터 × 수 FLOPs ≪ 수조 FLOPs)이라 제외

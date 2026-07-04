# RVSAC 반복횟수 증가에 따른 BPTT 안정성 및 TRM 대비 연산량 검토

## 0. 결론 요약

현재 로그의 `accuracy ~= 0.63-0.65`, `ce_last < ce_first`, 간헐적 `exact_accuracy > 0` 패턴은 붕괴라기보다 **재귀가 답을 개선하고 있으나 horizon 16에서 부분 정답률 plateau에 걸린 상태**로 보는 것이 타당하다. 과거 TRM 변형 실험에서 반복횟수 증가가 학습 경로를 바꿨다면, RVSAC에서도 `horizon` 증가 실험은 강하게 해볼 가치가 있다.

다만 RVSAC은 TRM보다 반복횟수 증가가 BPTT 안정성에 더 직접적으로 영향을 준다.

- TRM 기본 설정은 한 ACT step 안에서 21회 reasoning module forward를 하지만, 이 중 14회는 `no_grad`, 7회만 gradient-tracked이다. 또한 ACT step 사이 carry는 detach된다.
- RVSAC은 `horizon = H` 전체가 actor objective의 BPTT 경로다. 따라서 `H`를 늘리면 gradient chain 길이, activation memory, backward compute가 거의 선형으로 늘고, Jacobian product에 의한 폭주/소실 위험은 지수적으로 커질 수 있다.
- 현재 설정에서 `gamma = 0.95`이므로, residual transition Jacobian의 유효 norm이 대략 `1 / gamma = 1.0526`을 넘는 구간이 지속되면 discount가 Jacobian 증폭을 상쇄하지 못한다.

권장 실험은 바로 `H=64`로 가기보다 `H=24/32`부터 올리고, `u_h_ratio`만 보지 말고 `||d h_H / d h_t||` 또는 `A_t = I + d f / d h_t`의 power-iteration 기반 log-gain을 같이 기록하는 것이다.

## 1. 현재 로그 상태 해석

최근 구간:

```text
step 2800-3700:
accuracy        ~= 0.618-0.649
ce_first        ~= 1.10-1.16
ce_last         ~= 0.78-0.85
exact_accuracy  ~= 0.0-0.0156
u_h_ratio       ~= 0.175-0.193
td_abs          ~= 0.36-0.94
```

해석:

1. `ce_last < ce_first`가 안정적으로 유지되므로, 재귀 rollout은 실제로 prediction을 개선하고 있다.
2. `accuracy`가 65% 근처에서 흔들리는 것은 과거 실패 케이스와 유사한 plateau일 수 있다.
3. `exact_accuracy`가 step 3400 이후 간헐적으로 `0.0078`, `0.0156`을 찍는다. batch 128 기준 각각 1개, 2개 완전 정답이다. 즉 "풀 수 있는 샘플이 생기기 시작한" 상태다.
4. `u_h_ratio`가 0.17에서 0.19까지 올라가지만 폭주로 보이지는 않는다. 다만 이것은 update 크기/상태 크기 비율이지 Jacobian gain 자체가 아니므로 BPTT 안정성 진단으로는 부족하다.

따라서 현재 상태는 "학습 실패 확정"이 아니라 **horizon 16에서 부분적으로 개선되지만 충분한 iterative correction을 못 얻는 상태**에 가깝다.

## 2. RVSAC의 actor gradient

현재 구현은 다음 rollout을 사용한다.

```text
h_{t+1} = h_t + f_theta(h_t)
r_t     = -CE(C_theta(h_{t+1}), y)
J_H     = sum_{t=0}^{H-1} gamma^t r_t + gamma^H V_bar(h_H)
L_actor = - beta J_H
```

critic은 기본 설정 `critic_shapes_trunk = false`에서 `V_theta(sg[h_t])`만 학습하므로 trunk 안정성은 거의 actor BPTT가 결정한다.

편의상

```text
A_t = d h_{t+1} / d h_t = I + F_t
F_t = d f_theta(h_t) / d h_t
g_k = grad_{h_k} r(h_k)
v_H = grad_{h_H} V_bar(h_H)
```

라고 두면, 보상이 도착 상태 `h_{t+1}`에 정의되어 있으므로 costate는 다음처럼 흐른다.

```text
lambda_H = gamma^{H-1} g_H + gamma^H v_H
lambda_k = gamma^{k-1} g_k + A_k^T lambda_{k+1},  for k = H-1, ..., 1
```

transition parameter에 대한 actor gradient는

```text
grad_theta_f J_H
  = sum_{t=0}^{H-1} (d f_theta(h_t) / d theta_f)^T lambda_{t+1}
```

classifier parameter에는 직접 reward gradient가 간다.

```text
grad_theta_C J_H
  = sum_{t=0}^{H-1} gamma^t grad_theta_C r_t
```

중요한 부분은 `lambda` 재귀다. 전개하면 초기 step의 gradient에는 다음 Jacobian product가 들어간다.

```text
lambda_t contains
  (A_t^T A_{t+1}^T ... A_{k-1}^T) gamma^{k-1} g_k
```

따라서 bound는 대략 다음과 같다.

```text
||lambda_t||
  <= sum_{k=t}^{H-1} gamma^{k-1} (prod_{i=t}^{k-1} ||A_i||) ||g_k||
     + gamma^H (prod_{i=t}^{H-1} ||A_i||) ||v_H||
```

만약 평균적으로 `||A_i|| ~= rho`라면 주요 항은 `gamma^k rho^{k-t}` 꼴이다. 즉 안정 조건은 거칠게

```text
gamma * rho < 1
rho < 1 / gamma
```

이다. 현재 `gamma = 0.95`이므로

```text
1 / gamma = 1.0526
```

이다. residual update는 `A_t = I + F_t`라서 `F_t`가 작아도 `||A_t||`는 1 근방에서 시작한다. `rho`가 1.05를 조금만 넘어도 horizon을 늘렸을 때 gradient가 빠르게 커질 수 있다.

## 3. TRM과 BPTT 안정성 차이

TRM 기본 설정:

```text
H_cycles = 3
L_cycles = 6
halt_max_steps = 16
L_layers = 2
```

TRM의 한 ACT step 내부 reasoning module 호출 수는

```text
H_cycles * (L_cycles + 1)
= 3 * 7
= 21
```

하지만 구현상 gradient가 걸리는 것은 마지막 cycle뿐이다.

```text
no_grad calls       = (H_cycles - 1) * (L_cycles + 1) = 14
gradient calls      = L_cycles + 1 = 7
```

그리고 `new_carry = detach(z_H, z_L)`이므로 ACT step 사이로 BPTT가 이어지지 않는다. 따라서 TRM은 반복횟수 또는 `halt_max_steps`를 늘려도 **gradient chain 길이는 기본적으로 7 module calls에 고정**된다.

반면 RVSAC은

```text
gradient calls = horizon = H
```

이다. `H=16`이면 TRM의 gradient-tracked module depth 7보다 이미 길고, `H=32`면 약 4.6배의 activation-depth memory를 요구한다.

정리하면:

```text
TRM:
  긴 추론 = 많은 forward + truncated BPTT
  안정성 = carry detach와 no_grad burn-in에 의해 확보

RVSAC:
  긴 추론 = 긴 differentiable rollout
  안정성 = A_t = I + df/dh의 Jacobian product에 직접 의존
```

따라서 RVSAC이 반복횟수 증가로 성능 경로를 바꿀 가능성은 있지만, TRM처럼 공짜로 늘릴 수 있는 반복은 아니다.

## 4. 연산량 추산

현재 full 실험과 TRM attention 기본 설정을 기준으로 계산한다.

```text
Sudoku seq_len      = 81
puzzle_emb_len      = 16
S                   = 97
d                   = 512
num_heads           = 8
head_dim            = 64
L_layers            = 2
SwiGLU expansion    = 4
SwiGLU inter dim    = 1536
```

한 Transformer block의 대략적인 MAC은

```text
C_block
  ~= attention linear + attention matmul + SwiGLU MLP
  ~= 4 S d^2 + 2 S^2 d + 3 S d m
```

수치 대입:

```text
C_block ~= 0.340 GMAC
C_module = 2 blocks ~= 0.680 GMAC
```

RVSAC transition은 TRM module과 같은 2 blocks 뒤에 `out_proj: d -> d`가 하나 더 있다.

```text
C_out_proj = S d^2 ~= 0.025 GMAC
C_rvsac_step ~= 0.706 GMAC
```

LM head, value head, CE, TD(lambda)는 vocab 11 기준으로 block cost에 비해 작아서 여기서는 주요 비교에서 제외한다.

### 4.1 Forward-only

RVSAC:

```text
C_fwd_RVSAC(H) ~= H * 0.706 GMAC
```

TRM 한 ACT optimizer step:

```text
C_fwd_TRM_ACT ~= 21 * 0.680 = 14.29 GMAC
```

TRM max halt 16까지의 inference-style full solve:

```text
C_fwd_TRM_max16 ~= 16 * 21 * 0.680 = 228.61 GMAC
```

표:

| 설정 | forward GMAC/batch | TRM ACT step 대비 | TRM max16 대비 |
|---|---:|---:|---:|
| RVSAC H=8 | 5.65 | 0.40x | 0.025x |
| RVSAC H=16 | 11.29 | 0.79x | 0.049x |
| RVSAC H=32 | 22.59 | 1.58x | 0.099x |
| RVSAC H=64 | 45.17 | 3.16x | 0.198x |
| RVSAC H=128 | 90.35 | 6.32x | 0.395x |
| TRM ACT step | 14.29 | 1.00x | 0.063x |
| TRM max16 | 228.61 | 16.00x | 1.00x |

### 4.2 Forward + backward training compute

대략 backward를 differentiable forward의 2배로 잡으면:

```text
C_train ~= C_forward_all + 2 * C_forward_grad_tracked
```

RVSAC은 모든 horizon step이 gradient-tracked다.

```text
C_train_RVSAC(H)
  ~= H * C_rvsac_step + 2 H * C_rvsac_step
  ~= 3 H * 0.706 GMAC
```

TRM 한 ACT optimizer step은 21회 forward 중 7회만 gradient-tracked다.

```text
C_train_TRM_ACT
  ~= 21 C_module + 2 * 7 C_module
  ~= 35 C_module
  ~= 23.81 GMAC
```

TRM max halt 16까지 같은 샘플이 모두 진행된다고 보면:

```text
C_train_TRM_max16
  ~= 16 * 23.81
  ~= 381.02 GMAC
```

표:

| 설정 | train approx GMAC/batch | TRM ACT step 대비 | TRM max16 대비 |
|---|---:|---:|---:|
| RVSAC H=8 | 16.94 | 0.71x | 0.044x |
| RVSAC H=16 | 33.88 | 1.42x | 0.089x |
| RVSAC H=32 | 67.76 | 2.85x | 0.178x |
| RVSAC H=64 | 135.52 | 5.69x | 0.356x |
| RVSAC H=128 | 271.04 | 11.38x | 0.711x |
| TRM ACT step | 23.81 | 1.00x | 0.063x |
| TRM max16 | 381.02 | 16.00x | 1.00x |

주의: TRM의 "max16 train"은 16개의 ACT optimizer step으로 나뉜 truncated training이고, RVSAC은 한 optimizer step 안의 full BPTT다. 따라서 wall-clock throughput은 이론 MAC 외에 activation memory, kernel launch, compile, checkpointing 여부에 크게 좌우된다.

## 5. Activation memory와 BPTT depth

activation memory는 대략 gradient-tracked module call 수에 비례한다.

```text
TRM tracked depth       = 7 module calls
RVSAC tracked depth     = H module calls
memory ratio ~= H / 7
```

| RVSAC horizon | TRM tracked depth 대비 activation depth |
|---:|---:|
| 8 | 1.14x |
| 16 | 2.29x |
| 32 | 4.57x |
| 64 | 9.14x |
| 128 | 18.29x |

또한 RVSAC은 `states = [h_0, ..., h_H]`를 보관한다. 현재 batch 128, `S=97`, `d=512`, bf16 기준으로 state 하나는 대략

```text
128 * 97 * 512 * 2 bytes ~= 12.1 MiB
```

이다. `H=64`면 state tensor들만 약 780 MiB 수준이고, 실제 block activation은 이보다 훨씬 크다. 따라서 H를 크게 늘리면 compute보다 먼저 memory가 병목이 될 수 있다.

## 6. Parameter 차이

width는 동일하다.

```text
d = 512
num_heads = 8
L_layers = 2
expansion = 4
```

RVSAC은 TRM 대비 다음이 추가된다.

- transition `out_proj`: `d^2 = 262,144`
- value head: 약 `263,169`
- target value head: 약 `263,169`

TRM은 대신 q-head와 H/L init buffer가 있으나 매우 작다. 따라서 RVSAC은 TRM보다 대략

```text
~0.79M parameters
```

많다. 전체는 현재 Sudoku 단일 identifier 기준 대략:

```text
TRM   ~= 6.83M
RVSAC ~= 7.62M
```

정도다. ARC처럼 puzzle identifier 수가 늘면 양쪽에 같은 puzzle embedding이 추가되므로 차이는 거의 그대로 유지된다.

## 7. Horizon 증가 시 예상되는 실패 모드

### 7.1 Jacobian product 폭주

핵심 위험은 다음 항이다.

```text
prod_{i=t}^{k-1} (I + d f / d h_i)^T
```

현재 `u_h_ratio`는 0.19 수준으로 update 크기 자체는 작지만, 이것이 `||I + df/dh||`가 작다는 뜻은 아니다. residual identity 때문에 유효 Jacobian은 기본적으로 1 근처이며, 약간의 expansion만 있어도 long BPTT에서 누적된다.

위험 조건:

```text
mean log ||A_t|| > -log(gamma)
```

현재 `gamma=0.95`이므로

```text
-log(gamma) ~= 0.0513
```

즉 step당 log-gain이 0.05를 넘으면 discount가 장기 gradient 증폭을 못 막는다.

### 7.2 terminal value gradient의 장거리 전파

actor objective의 terminal 항:

```text
gamma^H V_bar(h_H)
```

은 value head parameter는 고정하지만 `h_H`로 gradient를 흘린다. 이 항은 좋은 costate 공급원이지만, horizon이 길어질수록 역시 모든 `A_t` product를 타고 초기 step까지 전파된다.

또한 `gamma=0.95`에서 terminal weight는 빠르게 작아진다.

```text
gamma^16  ~= 0.440
gamma^32  ~= 0.194
gamma^64  ~= 0.0375
gamma^128 ~= 0.0014
```

따라서 `H=64` 이상으로 가면 terminal bootstrap은 안정성 부담은 남기면서 의미 있는 장기 신호는 약해질 수 있다. long horizon 실험에서는 `gamma`를 같이 sweep해야 한다.

### 7.3 critic이 trunk를 직접 안정화하지 않음

현재 기본값은

```text
critic_shapes_trunk = false
```

이다. 이는 Phase 2/3류 라우팅 누수를 막는 데 안전하지만, value regression이 trunk에 contraction pressure를 주지는 않는다. 즉 long-horizon 안정성은 거의 actor objective와 transition parametrization에 달린다.

### 7.4 TRM과 달리 반복 증가가 곧 BPTT 증가

TRM은 반복횟수 증가가 대부분 forward-only compute 증가다. RVSAC은 반복횟수 증가가 곧 differentiable path 증가다. 따라서 TRM에서 "반복을 늘리니 갑자기 학습 경로가 바뀌었다"는 현상은 RVSAC에서도 가능하지만, 동일한 방식으로 무작정 늘리면 gradient 안정성 문제가 먼저 올 수 있다.

## 8. 권장 실험 순서

### 8.1 가장 먼저 할 sweep

동일 width, 동일 batch budget에서:

```text
H = 16, 24, 32
gamma = 0.95 유지
```

를 먼저 비교한다. `H=32`는 현재 H=16 대비 train compute 약 2배, activation depth 약 2배다. 아직 현실적인 첫 확장선이다.

이후 plateau가 깨지는 신호가 있으면:

```text
H = 48 or 64
gamma = 0.97, 0.98, 0.99 sweep
```

을 고려한다.

### 8.2 반드시 추가할 진단 지표

`u_h_ratio`만으로는 부족하다. 다음을 추가하는 것이 좋다.

```text
log_gain_t ~= log || (I + J_f(h_t)) v || / ||v||
```

또는 여러 step product:

```text
log || d h_{t+k} / d h_t ||
```

를 JVP/VJP power iteration으로 근사해 기록한다.

실험 판단 기준:

```text
mean log_gain < -log(gamma)  -> long BPTT가 비교적 안전
mean log_gain > -log(gamma)  -> horizon 증가 시 폭주 위험
```

### 8.3 H가 32를 넘으면 필요한 장치

1. activation checkpointing  
   메모리를 줄이는 대신 forward recomputation이 들어간다. H=64 이상에서는 사실상 필요할 가능성이 높다.

2. residual step scale  
   ```text
   h_{t+1} = h_t + alpha f(h_t)
   ```
   형태로 `alpha < 1` 또는 learned/gated alpha를 두면 `A_t = I + alpha J_f`가 되어 Jacobian gain을 낮출 수 있다.

3. transition norm regularization 또는 spectral control  
   완전한 spectral norm까지는 무겁더라도, power iteration으로 `J_f` gain penalty를 약하게 줄 수 있다.

4. truncated actor BPTT variant  
   rollout은 길게 하되 매 K step마다 detach하고 terminal value를 붙이는 방식:
   ```text
   long rollout H=64
   gradient segment K=16
   ```
   이는 TRM의 안정성 장치와 RVSAC objective를 절충하는 방법이다.

5. final/late reward weighting sweep  
   현재는 모든 step CE가 reward다. horizon을 늘릴 때 early-step CE가 objective를 지배하면 long correction이 덜 학습될 수 있다. late reward 가중을 키우는 ablation이 필요하다.

## 9. 내 판단

사용자의 가설, 즉 "65% 근처 plateau가 반복횟수 부족에서 오는 구조적 상한일 수 있다"는 해석은 현재 로그와 과거 TRM 변형 경험을 놓고 보면 충분히 그럴듯하다.

하지만 RVSAC에서 반복횟수 증가는 TRM보다 위험하다. TRM은 16 ACT step을 쓰더라도 gradient는 7 module depth로 잘려 있고, RVSAC은 `H` 전체가 BPTT다. 따라서 RVSAC의 horizon을 늘리는 실험은 다음 조건을 만족해야 한다.

```text
1. H=24/32부터 단계적으로 증가
2. Jacobian log-gain 진단 추가
3. H>=64에서는 checkpointing 또는 truncated BPTT 준비
4. gamma sweep 병행
5. exact_accuracy가 batch당 1-2개에서 안정적으로 증가하는지 확인
```

한 줄로 정리하면: **반복횟수를 늘리는 방향은 맞을 가능성이 크지만, RVSAC은 TRM처럼 forward 반복만 늘리는 구조가 아니라 full differentiable rollout이므로 `H` 증가가 곧 BPTT 안정성 문제다. H=32까지는 실험 가치가 높고, H=64 이상은 안정화 장치 없이 바로 가면 메모리와 Jacobian product가 먼저 병목이 될 가능성이 높다.**

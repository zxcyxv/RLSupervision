# RVSAC Carry-ACT와 Input Injection 검토

## 0. 핵심 요약

현재 RVSAC H=32 full-BPTT와 TRM은 같은 `optimizer step`이라는 말을 쓰지만, 그 step의 의미가 다르다.

```text
RVSAC H=32:
  매 optimizer step마다 새 batch 128개를 모두 사용
  각 샘플을 한 그래프 안에서 32-step rollout
  32-step BPTT gradient 1번으로 update

TRM:
  매 optimizer step마다 dataloader는 새 batch를 공급
  하지만 non-halted slot은 새 샘플을 버리고 이전 샘플을 계속 사용
  carry는 step 사이 detach되고, 매 ACT micro-step마다 update
```

따라서 RVSAC이 동일 optimizer step에서 TRM보다 많은 FLOPs와 많은 고유 샘플을 쓰는 것은 맞다. 반대로 고유 샘플 수를 맞추면, TRM의 평균 halt step 수만큼 RVSAC 쪽 FLOPs 해석이 유리해진다.

현재 H=32 RVSAC과 attention TRM의 optimizer-step FLOPs 비율을 대략 `2.85~2.95x`로 보면, TRM 평균 halt가 `m`일 때 고유 샘플 수 기준 RVSAC/TRM FLOPs 비율은:

```text
ratio_sample_matched ~= 2.95 / m

m=4:   0.74x
m=8:   0.37x
m=16:  0.18x
```

하지만 동일 optimizer step에서도 RVSAC이 TRM보다 낮은 exact accuracy를 보인다면, 그것은 최적화 효율 측면에서 분명한 약점이다. 공정 비교를 위해서는 RVSAC에도 TRM식 carry-ACT ablation을 둬야 한다.

## 1. TRM의 샘플 소비 방식

TRM batch는 `B`개의 persistent slot으로 볼 수 있다. 각 slot은 현재 풀고 있는 샘플과 carry를 가진다.

코드 의미는 다음과 같다.

```text
if carry.halted at start of step:
  use dataloader batch sample
else:
  ignore dataloader batch sample
  keep previous current_data
```

slot 하나에서 step 4에 halt가 난다고 하면:

```text
optimizer step | dataloader sample | slot이 실제 사용한 sample | step 끝 halted
-------------- | ----------------- | ------------------------- | ---------------
1              | A                 | A                         | False
2              | B                 | A                         | False
3              | C                 | A                         | False
4              | D                 | A                         | True
5              | E                 | E                         | ...
```

중요한 점은 step 4에서 halt가 나도 step 4의 dataloader sample `D`를 쓰지 않는다는 것이다. halt 결정은 그 step 끝에 만들어지고, 다음 step부터 새 샘플을 받는다.

평균 halt step이 `m`이면, 같은 optimizer step 수 `N`, batch size `B`에서 실제 고유 샘플 episode 수는:

```text
RVSAC unique presentations ~= N * B
TRM unique episodes        ~= N * B / m
```

즉 평균 halt가 8이면 TRM은 RVSAC 대비 고유 샘플을 약 1/8만 실제로 사용한다.

## 2. 한 optimizer step의 gradient 구성

### 2.1 RVSAC H=32 full

```text
h0 = embed(x)
h_{t+1} = h_t + f_theta(h_t),  t=0..31
r_t = -CE(C_theta(h_{t+1}), y)
J = sum_t gamma^t r_t + gamma^32 V_bar(h_32)
```

transition gradient는 32-step costate를 가진다.

```text
dJ/dtheta_f = sum_t (df_theta(h_t)/dtheta_f)^T lambda_{t+1}
lambda_t = local_t + (I + df/dh_t)^T lambda_{t+1}
```

특징:

```text
transition: 32-step future reward/value에서 오는 full BPTT gradient
lm_head:    모든 step CE_1..CE_32에서 직접 gradient
embedding:  h0를 통해 32-step 전체 gradient
value_head: critic loss에서만 gradient, 기본적으로 trunk에는 sg[h_t] 때문에 안 감
V_bar:      parameter gradient 없음, terminal에서는 h_H로만 gradient
```

### 2.2 RVSAC H=32, K=8 segmented

```text
segment 0: h0  -> h8
detach
segment 1: h8  -> h16
detach
segment 2: h16 -> h24
detach
segment 3: h24 -> h32
```

forward FLOPs는 H=32 full과 거의 같다. 그러나 gradient chain은 최대 8로 잘린다.

```text
transition: 4개의 독립적인 8-step BPTT gradient
embedding:  segment 0의 직접 gradient만 받음
long credit: segment terminal V_bar bootstrap에 의존
```

### 2.3 TRM

TRM의 한 optimizer step은 ACT 전체 16회를 BPTT하지 않는다. 한 ACT micro-step만 학습한다.

현재 attention TRM 설정:

```text
H_cycles=3, L_cycles=6, L_layers=2

한 ACT micro-step 내부:
  L_level calls total = 3 * (6 + 1) = 21
  no_grad calls       = 2 * 7 = 14
  grad calls          = 7
```

loss:

```text
L_TRM = CE(C(z_H), y) + 0.5 * BCE(q_halt(z_H), exact_correct)
```

특징:

```text
trunk:          마지막 grad cycle만 gradient
lm_head:        현재 logits CE에서 gradient
q_head:         halt BCE에서 gradient
embedding:      grad cycle의 input injection 경로로 gradient
previous carry: detach되어 이전 ACT step으로 gradient 안 감
no_grad burn-in: forward state만 만들고 parameter gradient 없음
```

즉 TRM은 한 샘플을 여러 optimizer step 동안 반복하지만, 그것은 16-step BPTT가 아니라 16개의 local update다.

## 3. RVSAC Carry-ACT 제안

TRM과 FLOPs/샘플 소비 구조를 맞추려면 RVSAC도 carry slot을 가져야 한다. 단순히 H=32 full rollout의 batch를 줄이는 것은 optimizer update 수도 같이 줄이므로 TRM과 같은 구조가 아니다.

제안 구조:

```text
K = 8                  # optimizer step당 BPTT 길이
max_segments = 4       # 최대 reasoning depth = 32

carry:
  h: current hidden state
  segments_or_depth: current reasoning depth
  halted: slot별 새 샘플 수용 여부
  current_data: slot별 현재 sample/label
```

한 optimizer step:

```text
1. halted slot에는 dataloader의 새 샘플을 넣고 h=embed(x)로 reset
2. non-halted slot은 dataloader 샘플을 버리고 이전 current_data/h 유지
3. 현재 샘플에 대해 K=8 rollout
4. segment loss로 backward/update
5. h_K.detach()를 carry에 저장
6. exact solved, max depth, 또는 learned/value halt면 halted=True
```

slot 하나의 예:

```text
step 1:
  새 샘플 A
  A: h0 -> h8
  못 풀면 keep A

step 2:
  dataloader B는 버림
  A: h8 -> h16

step 3:
  C 버림
  A: h16 -> h24

step 4:
  D 버림
  A: h24 -> h32
  solved 또는 max depth로 halt

step 5:
  새 샘플 E
```

이 구조는 현재 `H=32,K=8`과 다르다.

```text
현재 H=32,K=8:
  한 optimizer step 안에서 4 segment를 전부 처리
  매 step 새 샘플 128개를 모두 사용

Carry-ACT K=8:
  한 optimizer step에서 segment 1개만 처리
  non-halted slot은 같은 샘플 유지
  평균 segment 수만큼 고유 샘플 소비가 줄어듦
```

segment loss는 기존 RVSAC의 K-step actor-critic을 그대로 쓸 수 있다.

```text
for i=0..K-1:
  h_{i+1} = h_i + f(h_i)
  r_i = -CE(C(h_{i+1}), y)

critic:
  TD(lambda) over K steps with bootstrap V_bar(h_K)

actor:
  J_segment = sum_i gamma^i r_i + gamma^K V_bar(h_K)
```

초기 halt rule은 학습 안정성 확인용으로 단순하게 둘 수 있다.

```text
halt = exact_correct_at_h_K or depth >= H_max
```

실제 inference까지 고려하면 label-based exact halt는 사용할 수 없으므로, 이후에는 value-based halt 또는 halt head가 필요하다.

## 4. FLOPs 해석

현재 H=32 full RVSAC:

```text
한 optimizer step:
  32 transition calls
  L_layers=2이므로 64 transformer block calls
  모두 gradient-tracked
```

Carry-ACT K=8:

```text
한 optimizer step:
  8 transition calls
  L_layers=2이므로 16 transformer block calls
  모두 gradient-tracked
```

따라서 step당 FLOPs는 대략 H=32 full의 1/4이다. 평균 4 segments를 쓰면 샘플 하나당 총 reasoning depth는 32로 같지만, optimizer update는 4번으로 나뉘고 BPTT는 K=8에서 끊긴다.

이 실험의 의미:

```text
Carry-ACT RVSAC도 잘 되면:
  full 32-step BPTT보다 recursive state/value 구조와 sample reuse가 핵심일 가능성
  FLOPs 효율 개선 가능

Carry-ACT RVSAC이 65% plateau로 돌아가면:
  현재 H=32 full의 성능 점프는 long BPTT credit assignment가 핵심일 가능성
```

## 5. Input Injection에 대한 검토

TRM은 input injection이 사실상 필수다.

```text
TRM:
  z_H, z_L 초기값이 learned/constant buffer
  loop 바깥 residual state가 input별로 초기화되지 않음
  input injection이 없으면 모든 입력이 같은 hidden trajectory로 붕괴하기 쉬움
```

RVSAC은 다르다.

```text
RVSAC:
  h0 = embed(input) + puzzle_emb
  h_{t+1} = h_t + f(h_t)
```

입력별 차이 `Delta_t = h_t(x) - h_t(x')`는:

```text
Delta_{t+1} ~= (I + J_f(h_t)) Delta_t
```

따라서 RVSAC은 원리적으로 input injection 없이도 입력별 state를 유지할 수 있다. 그러나 `I + J_f`가 입력 관련 subspace에서 contraction이면 입력 정보가 rollout 중 희석될 수 있다. residual memory가 항상 안전한 input anchor라는 보장은 없다.

input injection을 넣으면:

```text
h_{t+1} = h_t + f(h_t + alpha e)
e = embed(input)
```

입력 차이는 대략:

```text
Delta_{t+1} ~= A_t Delta_t + B_t Delta e
```

즉 recurrent state가 contraction이어도 input difference가 매 step 다시 공급된다.

## 6. Gradient 폭주 관점

피해야 할 형태:

```text
h_{t+1} = h_t + alpha e + f(h_t)
```

이 경우 `e`가 state에 직접 누적된다.

```text
h_H = h_0 + H * alpha e + sum_t f(h_t)
```

forward scale과 embedding gradient가 H배로 커질 수 있다. 예전 TRM 변형에서 loop 바깥 residual을 넣었을 때 폭주한 실패와 같은 계열 위험이 있다.

더 안전한 형태:

```text
u_t = transition(h_t + alpha e)
h_{t+1} = h_t + u_t
```

현재 `transition()`은 내부 첫 단계에서 `rms_norm`을 적용한다.

```text
u_t = out_proj(Blocks(rms_norm(h_t + alpha e)))
```

이때 recurrent Jacobian은:

```text
A_t = I + J_f(rms_norm(h_t + alpha e)) * J_rms(h_t + alpha e)
```

즉 폭주의 핵심인 `prod_t A_t` 구조는 여전히 존재하지만, input injection이 state에 직접 누적되지는 않는다. RMSNorm은 절대 scale 증가를 억제한다.

embedding 쪽 gradient는 모든 step에서 추가로 들어간다.

```text
dL/de = dL/dh0 + alpha * sum_t J_f^T J_rms^T lambda_{t+1}
```

따라서 `alpha=1`은 금지는 아니지만 강할 수 있다. 첫 실험은 다음 순서가 안전하다.

```text
alpha = 0.25
alpha = 0.5
alpha = 1.0
```

또는 learnable scalar를 작게 초기화한다.

```text
alpha init ~= 0 or 0.25
```

## 7. K=8 Segment와 Input Injection의 결합

K=8 segment를 TRM의 평균 halt step 8에 대응시키려면, input injection은 segment local solver의 anchor 역할을 할 수 있다.

후보 A: 매 step injection

```text
for t in segment:
  u_t = transition(h_t + alpha e)
  h_{t+1} = h_t + u_t
```

TRM은 reasoning module 호출마다 input injection을 하므로, 이쪽이 TRM에 더 가깝다.

후보 B: segment 시작에서만 injection

```text
if t % K == 0:
  u_t = transition(h_t + alpha e)
else:
  u_t = transition(h_t)
```

이는 segment 시작마다 입력 조건을 재고정하고, segment 내부 8 step은 autonomous refinement로 두는 방식이다. 더 보수적이지만 anchor가 약할 수 있다.

중요한 점: segment boundary detach 때문에 `h` 경로의 gradient는 K=8에서 끊긴다. 그러나 같은 `e`가 모든 segment에서 쓰이면, input embedding은 각 segment에서 gradient를 받는다.

```text
dL/de = sum_segments sum_i alpha J_i^T lambda_{s,i+1}
```

이 경로가 너무 강하면 embedding이 흔들릴 수 있으므로 scale을 작게 시작해야 한다.

## 8. 필요한 진단 지표

TRM sample reuse:

```text
new_sample_fraction = halted_start.mean()
estimated_mean_halt = 1 / new_sample_fraction
used_unique_samples += halted_start.sum()
```

RVSAC rollout:

```text
accuracy@8, @16, @24, @32
exact@8, @16, @24, @32
oracle_exact_any_step
earliest_solved_step
given_cell_accuracy_by_step
```

input drift/collapse:

```text
cosine(h_t, h_0)
||h_t - h_0|| / ||h_0||
pairwise ||h_t(x)-h_t(x')|| over t
linear probe: recover original input/given mask from h_t
```

stability:

```text
grad_norm
u_h_ratio
critic_loss / td_abs
value_mean / eq_gap
mean log ||I + df/dh||
```

비교 곡선:

```text
exact vs optimizer steps
exact vs estimated FLOPs
exact vs estimated unique samples
exact vs estimated unique samples * FLOPs
```

## 9. 추천 실험 순서

1. TRM에 실제 sample reuse 지표 추가

```text
new_sample_fraction
estimated_mean_halt
used_unique_samples
```

2. RVSAC H=32 full baseline 유지

```text
arch.horizon=32
arch.bptt_segment=0
arch.loss.gamma=0.975
```

3. RVSAC H=24 중간점

```text
arch.horizon=24
arch.bptt_segment=0
arch.loss.gamma=0.966
```

4. RVSAC input injection ablation

```text
arch.input_injection=transition
arch.input_injection_scale=0.25
arch.input_injection_scale=0.5
arch.input_injection_scale=1.0
```

5. RVSAC Carry-ACT K=8

```text
arch.carry_act=True
arch.act_segment=8
arch.halt_max_segments=4
arch.input_injection=transition
arch.input_injection_scale=0.25
```

이 실험이 TRM과 가장 가까운 공정 비교다.

## 10. 결론

현재 RVSAC H=32 full은 TRM 대비 optimizer-step FLOPs가 크고, 같은 step에서 성능이 밀리면 최적화 효율상 불리하다. 그러나 TRM은 non-halted slot에서 dataloader 샘플을 버리므로, 고유 샘플 수 기준 비교는 단순 step 비교와 다르다.

공정한 구조 비교를 위해서는 RVSAC에 TRM식 carry-ACT ablation을 추가해야 한다. 이때 K=8 segment를 ACT micro-step으로 삼고, slot별 sample carry를 유지하면 샘플 소비와 adaptive compute 구조가 TRM과 맞춰진다.

Input injection은 RVSAC에 원리적으로 필수는 아니지만, long rollout에서 입력 anchor가 약해지는 문제를 완화할 수 있다. 폭주 위험을 피하려면 `e`를 state에 직접 누적하지 말고, `transition(h_t + alpha e)` 형태로 넣어야 한다. 첫 실험은 `alpha=0.25` 또는 `0.5`가 안전하다.

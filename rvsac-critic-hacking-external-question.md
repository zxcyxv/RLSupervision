# 다른 AI에게 물어볼 질문 — Critic Exploitation / Value 폭주 문제

`rvsac-critic-hacking-diagnosis-and-fixes.md`에서 진단한 문제를 다른 AI(외부 시각)에게
물어보기 위한 자기완결적 프롬프트. 이 대화 맥락을 전혀 모르는 상태에서 바로 이해하고
답할 수 있도록 구조·증상·원인 분석까지 압축해서 담았다.

## 사용 팁

- **아래 프롬프트를 그대로 복사**해서 다른 AI 세션에 붙여넣으면 된다.
- 자기완결성이 핵심 — 구조·수식·증상·원인 분석까지 다 넣어야 "그건 이미 아시는 대로입니다"
  같은 헛도는 답을 안 받는다.
- 문헌 회상(논문 이름, 표준 기법명)이 중요하므로 **영어로 물어보면 더 정확한 답을 받을
  확률이 높다** — RL 논문 용어 대부분이 영어 원어이기 때문. 아래 한글 버전 다음에 영어
  버전도 같이 준비해뒀다.
- "twin critic", "PopArt", "symlog"는 이미 검토한 후보 해법들인데, 다른 AI가 이 셋 외에
  놓친 표준 기법을 아는지 확인하려고 일부러 넣었다.

---

## 프롬프트 (한글)

```
재귀형 신경망 하나로 정책(actor)과 가치함수(critic)를 동시에 학습시키는 구조를 만들었습니다.

구조:
h_{t+1} = h_t + f_θ(h_t)          (t=0..H-1, 잔차 재귀)
r_t = -CE(분류기(h_{t+1}))         (화이트박스 보상, 자기 자신의 예측 오차)
V_θ(h)                            (스칼라 가치 헤드)
목적함수: L = TD(λ) 비평가 손실 - β·(Σγ^t r_t + γ^H·V_bar(h_H))
액터 그래디언트는 V_bar(h_H)(타깃망, 파라미터 동결, h_H로만 grad 흐름)를 통해서만 흐름 —
DDPG의 actor 업데이트(∇_θμ(h)·∇_aQ(h,a))와 동일 구조.

관측된 문제: r_t≤0이 항상 성립하므로(CE≥0) 진짜 V는 항상 ≤0이어야 하는데, 학습 중 V가
이론상 불가능한 양수(최대 +20 근처)까지 폭주. 원인 분석 결과:

1. actor가 매 스텝 "critic이 지금 후하게 평가하는 방향"으로 h_H를 미는데, critic의 근사
   오차가 진짜 개선인지 구분을 안 함(critic exploitation).
2. γ가 1에 가까울 때(0.975), V 전체에 상수를 더해도 TD 잔차가 (1-γ)배만큼만 늘어나는
   "거의 평평한" 방향이 존재해서 critic이 잘못된 상수 오프셋에 안착해도 손실이 낮게 유지됨.
3. TD(λ) 재귀의 마지막 지점(h_H, 부트스트랩의 base case)은 λ가 뭐든 항상 순수 1-step
   TD라 교정력이 제일 약하고, 이 입력 자체가 critic 자신의 회귀 훈련 세트에도 안 들어있어
   (순수 외삽) actor가 정확히 이 지점을 공격함.

질문:
- MuZero, Dreamer(v1~v3), DDPG/TD3/SAC, AlphaZero류 self-play 시스템 등 비슷하게 "자기
  자신의 예측/보상을 기반으로 학습하는 actor-critic" 구조에서 이런 종류의 실패(critic이
  자기참조적으로 부풀거나, actor가 critic 근사 오차를 타고 exploit하는 것)가 실제로
  어떻게 나타나고, 각 시스템은 이를 막기 위해 구체적으로 뭘 하나요? 
- 이 문제를 다룬 표준 논문이나 이론적 이름(예: "value overestimation bias", "actor
  exploiting critic error")이 있다면 알려주세요.
- γ가 1에 가까울 때 생기는 "상수 오프셋에 둔감해지는" 문제를 다른 문헌에서는 어떻게
  부르고 어떻게 해결하나요(예: average-reward RL의 차등 가치함수 논의와 관련 있는지)?
```

---

## Prompt (English)

```
I built an architecture where a single recursive network is trained as both a policy
(actor) and a value function (critic) simultaneously.

Architecture:
h_{t+1} = h_t + f_θ(h_t)          (t=0..H-1, residual recursion)
r_t = -CE(classifier(h_{t+1}))     (white-box reward, the network's own prediction error)
V_θ(h)                            (scalar value head)
Objective: L = TD(λ) critic loss - β·(Σγ^t r_t + γ^H·V_bar(h_H))
The actor's gradient flows only through V_bar(h_H) (a target network, frozen params,
gradient flows only through h_H) — structurally identical to DDPG's actor update rule
(∇_θμ(h)·∇_aQ(h,a)).

Observed problem: since r_t≤0 always holds (CE≥0), the true V should always be ≤0, but
during training V blows up to theoretically-impossible positive values (up to ~+20).
Root-cause analysis found:

1. Every step, the actor pushes h_H toward "whatever direction the critic currently
   rates generously," without distinguishing genuine improvement from approximation
   error (critic exploitation).
2. When γ is close to 1 (0.975), there exists a nearly-flat direction where adding a
   constant to V everywhere only increases the TD residual by a factor of (1-γ) — so
   the critic can settle on a wrong constant offset while keeping loss low.
3. The final point of the TD(λ) recursion (h_H, the bootstrap's base case) is always
   pure 1-step TD regardless of λ, giving it the weakest correction of any point in the
   rollout — and this exact input is never part of the critic's own regression training
   set (pure extrapolation via the target network). The actor's gradient targets exactly
   this point.

Questions:
- In comparable "actor-critic trained on its own predictions/reward" systems — MuZero,
  Dreamer (v1-v3), DDPG/TD3/SAC, AlphaZero-style self-play — how does this class of
  failure (critic self-referentially inflating, or the actor exploiting the critic's
  approximation error) actually manifest, and what does each system concretely do to
  prevent it? (twin critics/min, delayed target networks, PopArt/return normalization,
  symlog value transforms, reward/value clipping, etc.)
- Are there standard papers or named concepts for this (e.g. "value overestimation
  bias," "actor exploiting critic error") I should look up?
- How does the literature describe/address the "insensitivity to a constant offset"
  problem that appears as γ approaches 1 (e.g. is this related to the differential
  value function discussion in average-reward RL)?
```

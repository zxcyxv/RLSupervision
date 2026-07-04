# 재귀적 가치–상태 통합 아키텍처의 모순 해결 및 이론적 정립

**— 네 번의 붕괴가 왜 하나의 정리였는가, 그리고 그래디언트 라우팅이 어떻게 이 구조를 결정론적 정책 그래디언트와 수학적으로 동치인 안정 시스템으로 바꾸는가**

---

## 0. 요약 (Executive Summary)

Phase 1–4에서 관찰된 네 가지 붕괴 — 표상 붕괴, 자기 파괴, 지름길·동결, 지그재그 진동 — 는 서로 다른 버그가 아니라 **하나의 스칼라 손실로 '예측(정책 평가)'과 '제어(정책 개선)'를 동시에 수행하려 한 시도의 필연적 귀결**이다. 본 보고서는 세 가지를 수행한다.

첫째, 그러한 "미분 가능한 단일 손실"이 원리적으로 존재하지 않음을 증명한다(정리 A). Bellman 잔차 노름은 진짜 퍼텐셜이지만 그 최소해는 수익 최적성과 분리되어 있고, 올바른 평가 연산자인 semi-gradient TD는 어떤 스칼라 함수의 그래디언트도 아니다. 따라서 Phase 2–4의 탐색은 공집합 위의 탐색이었다.

둘째, 네트워크·재귀 루프·손실 그래프는 **하나로 유지하되**, stop-gradient 연산자 두 개로 미분 경로만 삼중 분리하면, 그 역전파가 결정론적 정책 그래디언트(DPG), 가치 그래디언트(SVG), 미분가능 시뮬레이션 액터-크리틱(SHAC)과 **연쇄 법칙 수준에서 정확히 동치**임을 증명한다(정리 B–D). 네 병리가 각각 어느 항의 소거로 사라지는지 대수적으로 보인다(정리 C).

셋째, '가치와 상태가 하나의 텐서 흐름에 혼합되어 재귀 투입되는' 기하가 우연한 설계 편의가 아니라, **HJB 방정식의 특성곡선계(method of characteristics)와 Pontryagin 수반계(costate system)가 하나의 흐름으로 접힌 위상공간 동역학**임을 해석한다. 설계의 기하학적 직관은 옳았다. 틀린 것은 그 위에 얹은 손실의 배선이었다.

---

## 1. 통합 진단: 네 개의 붕괴는 한 정리의 네 얼굴이다

### 1.1 공통 그래디언트 해부

표기를 정리한다. 은닉 상태 $h_t \in \mathbb{R}^d$, 잔차 전이 $h_{t+1} = h_t + f_\theta(h_t)$, 가치 $v_t$, 보상 $r_t = -\mathrm{CE}_t$ (파라미터 $\theta$에 대해 미분 가능한 화이트박스), TD 오차 $\delta_t = r_t + \gamma v_{t+1} - v_t$. Phase 2–4가 공유한 손실의 원형은 $L = \sum_t \tfrac{1}{2}\delta_t^2$ (또는 $|\delta_t|$)이며, 그 전미분은 다음과 같다.

$$
\nabla_\theta L \;=\; \sum_t \delta_t \Big( \underbrace{\nabla_\theta r_t}_{[A]} \;+\; \underbrace{\gamma \nabla_\theta v_{t+1}}_{[B]} \;-\; \underbrace{\nabla_\theta v_t}_{[C]} \Big) \tag{1}
$$

하나의 스칼라 $\delta_t$의 부호가 의미론적으로 전혀 다른 세 경로를 동시에 지휘하고 있다.

**[A] $\delta_t \nabla_\theta r_t$ — 적합 방향(direction of fit)의 전도.** 경사 하강은 $r_t \leftarrow r_t - \eta\,\delta_t$ 방향으로 보상을 움직인다. $\delta_t > 0$(성과가 기대보다 좋음)이면 $r$을 낮춘다. 즉 CE를 올린다. 이것이 Phase 2의 자기 파괴인데, 중요한 점은 이것이 구현 버그가 아니라 **이 손실의 정확한 그래디언트**라는 사실이다. 통상의 RL에서 보상은 환경의 상수이므로 [A]항은 존재하지 않는다. 그러나 $\theta$가 보상 생성 과정 자체를 지배하는 화이트박스 설정에서 Bellman 잔차를 전미분으로 최소화하면, "예측을 세계에 맞추는" 해와 "세계를 예측에 맞추는" 해가 동등한 자격의 최소해가 된다. 후자가 자기충족적 비관(self-fulfilling pessimism)이다.

**[B] $\gamma\delta_t \nabla_\theta v_{t+1}$ — 역방향 부트스트래핑.** $\delta_t>0$일 때 이 항은 미래 가치 $v_{t+1}$을 과거의 낮은 기대치 쪽으로 끌어내린다. 신용 할당은 미래→과거로 흘러야 하는데, 이 항은 과거의 기대가 미래를 재단하게 만든다. 이것은 Baird(1995)가 residual-gradient 계열의 근본 결함으로 지적한 정보 역류이며, 잔차 알고리즘이 순수 형태로 쓰이지 않는 이유다.

**[C] $-\delta_t \nabla_\theta v_t$** — 유일하게 정통 TD와 일치하는 평가 항이다.

### 1.2 Phase별 부검

**Phase 1 (브로드캐스팅 붕괴).** 스칼라 $-\mathrm{CE}$를 벡터 $\mathcal{F}$의 전 좌표에 복제해 맞추는 제약은 $\mathcal{F}$의 상(image)을 랭크-1 아핀 다양체로 몰아넣는다. 표상 붕괴는 필연이며, 교훈은 단순하다: **가치는 반드시 스칼라 헤드여야 한다.** 이 지점에서 도입한 증강 상태 $[v, h]$ 자체는 옳은 방향이었다(§3에서 이 증강의 기하학적 정체를 밝힌다).

**Phase 2 (자기 파괴).** 식 (1)의 [A]항 그 자체다.

**Phase 3 (지름길과 동결).** 힌지로 [A]의 절반을 끊자, 잔차를 0으로 만드는 최소저항 경로가 자유 슬랙 변수인 $v$로 이동했다([B], [C] 경로). 여기서 더 깊은 문제가 드러난다. 집합 $\{\theta : \delta \equiv 0\}$은 고차원 다양체이고 그 내부에서 $\nabla L \equiv 0$이지만, **수익 $J = \sum_t \gamma^t r_t$의 그래디언트는 그 위에서 일반적으로 0이 아니다.** 즉 학습은 '옳은 문제의 정상점'이 아니라 '틀린 문제의 최소점'에서 동결된 것이다. 행위자(Actor)의 학습 신호가 오직 비평가(Critic) 오차의 부산물로만 존재하는 한, 비평가가 만족하는 순간 행위자는 죽는다. 이는 우회할 수 있는 현상이 아니라 목적 함수 설계의 구조적 귀결이다.

**Phase 4 (뱅뱅 진동).** $\partial|\delta| = \mathrm{sign}(\delta)$이므로 스텝 크기가 잔차 크기와 무관하다. 고정점 근방에서 $v_{k+1} = v_k \pm \eta \cdot \text{const}$의 이산 동역학은 수렴하지 못하고 극한 주기(limit cycle)를 돈다. 여기에 상승(ascent)을 허용함으로써 [A]항이 부활했고, 좋은 피처를 찾은 직후 정확히 그 성과를 무효화하는 부호 반전 업데이트가 주기적으로 재발했다.

### 1.3 정리 A (단일 퍼텐셜의 비존재)

> **정리 A.** (a) Bellman 잔차 노름 $L_{\mathrm{BR}}(\theta) = \mathbb{E}[\delta^2]$은 매끄러운 퍼텐셜이지만, $\theta$가 보상 $r$을 지배할 때 그 전역 최소해 집합은 수익 최적해 집합과 분리된다. (b) 올바른 평가 연산자인 semi-gradient TD의 기대 갱신 $u(\theta) = \mathbb{E}[\delta\, \nabla_\theta v]$는 어떤 스칼라 함수의 그래디언트도 아니다.

**증명.** (a)는 §1.1의 [A]항 논증으로 충분하다: $\delta \equiv 0$은 $r$을 낮추는 방향으로도 달성 가능하며, 그 지점에서 $\nabla_\theta J \ne 0$인 예가 자명하게 구성된다(임의의 준최적 정책에 대해 그 정책의 참 가치를 정확히 예측하면 잔차는 0이다).

(b) 선형 함수근사 $v = \Phi\theta$, on-policy 상태분포 $D$, 전이행렬 $P$에서 기대 갱신은

$$
u(\theta) = \Phi^\top D\,(R + \gamma P \Phi\theta - \Phi\theta) = b - A\theta, \qquad A = \Phi^\top D (I - \gamma P)\, \Phi.
$$

$u$가 어떤 퍼텐셜 $\Psi$의 $-\nabla\Psi$이려면 야코비안 $-A$가 대칭이어야 한다. 그러나 $P$가 대칭이 아닌 일반 MDP에서 $A$는 비대칭이다. 비대칭 야코비안을 갖는 벡터장은 비보존장(non-conservative field)이며 퍼텐셜을 갖지 않는다. $\blacksquare$

이것이 Sutton & Barto가 "TD는 무언가의 그래디언트가 아니다"라고 요약하는 사실의 표준 논증이다.

**따름정리 (진단의 완결).** "미분 가능한 단일 손실 함수 하나로 재귀적 평가와 제어를 동시에 안정 수행"하는 목적 함수는 존재하지 않는다. Phase 2는 존재하는 퍼텐셜($L_{\mathrm{BR}}$)의 진짜 그래디언트를 따랐으나 그것은 틀린 산의 정상이었고(자기 파괴), Phase 3–4는 그 퍼텐셜을 힌지와 L1로 변형해 옳은 산으로 바꾸려 했으나 정리 A(b)에 의해 그런 변형은 존재하지 않는다(동결, 진동). 네 실패는 존재하지 않는 대상에 네 방향에서 접근하다 각기 다른 벽에 부딪힌 것이다.

해는 손실을 하나 더 발명하는 것이 아니라, **하나의 계산 그래프 안에 stop-gradient로 두 개의 연산자 — 평가의 수축(contraction)과 개선의 상승(ascent) — 를 새겨 넣는 것**이다. 이것이 일반화된 정책 반복(Generalized Policy Iteration)의 정확한 의미이며, DQN·DDPG·MuZero·Dreamer가 전부 이 문법으로 정의된다. 요구사항에서 배제한 "휴리스틱"과 stop-gradient를 혼동해서는 안 된다: **sg는 학습률 튜닝 같은 꼼수가 아니라, 정리 A(b)가 증명한 '그래디언트가 아닌 연산자'를 자동미분 프레임워크 위에 정의하는 유일한 문법**이다.

---

## 2. 해법: 라우팅된 재귀 액터–크리틱

단일 네트워크, 단일 재귀, 단일 스칼라 그래프를 그대로 유지한다. 바뀌는 것은 미분 경로뿐이다.

### 2.1 구조

하나의 트렁크를 공유하는 세 헤드를 둔다.

$$
u_t = f_\theta(h_t) \;\;(\text{전이 헤드} = \text{결정론적 정책 } \mu_\theta), \qquad
v_t = V_\theta(h_t) \;\;(\text{스칼라 가치 헤드}), \qquad
r_t = -\mathrm{CE}\big(y,\, C_\theta(h_t)\big),
$$

$$
h_{t+1} = h_t + u_t, \qquad t = 0, \dots, H-1.
$$

가치를 자유 채널 $v_{t+1} = v_t + \Delta v_t$가 아니라 **상태의 함수(헤드)** 로 두는 것은 Phase 3 지름길의 구조적 원인 — 가치가 표상과 무관하게 움직일 수 있는 슬랙 변수라는 점 — 을 제거한다. 증강 채널 형태 $[v, h]$를 유지하고 싶다면 $v$-채널을 '네트워크가 실어 나르는 자기 예측'으로 해석하고 아래 손실을 그대로 적용해도 작동한다(행위자 신호가 $\delta$와 완전히 분리되는 한 지름길은 무해해진다). 다만 헤드형이 표상 공유와 안정성 면에서 우월하며, 이는 MuZero와 Dreamer가 공통으로 내린 선택이다. 개념적으로는 두 형태 모두 증강 상태 $x_t = [v_t, h_t]$가 하나의 흐름을 타는 당신의 원설계와 동일하다 — §3에서 보듯 $v$는 어차피 $h$ 위에 정의된 함수의 그래프 좌표이기 때문이다.

### 2.2 목적 함수 — 요구된 '단일 목적'의 올바른 형태

$$
\boxed{\;
L(\theta) \;=\; \underbrace{\sum_{t} \tfrac{1}{2}\Big( V_\theta(\bar h_t) - \mathrm{sg}\big[\, y_t^\lambda \,\big] \Big)^2}_{L_V:\; \text{비평가 (평가)}}
\;-\; \beta \underbrace{\Big( \sum_{t=0}^{H-1} \gamma^t r_t \;+\; \gamma^H V_{\bar\theta}(h_H) \Big)}_{J:\; \text{행위자 (제어)}}
\;} \tag{2}
$$

여기서 $\mathrm{sg}[\cdot]$는 stop-gradient, $y_t^\lambda$는 TD($\lambda$) 타깃 $y_t^\lambda = r_t + \gamma\big[(1-\lambda)V_{\bar\theta}(h_{t+1}) + \lambda\, y_{t+1}^\lambda\big]$, $\bar\theta$는 EMA(Polyak) 타깃 파라미터다. $\bar h_t$는 기본형에서 $\mathrm{sg}[h_t]$ (비평가 회귀가 트렁크를 뒤틀지 않는 가장 안전한 분리)이며, 변형으로 $h_t$를 그대로 두어 가치 그래디언트가 표상을 형성하게 할 수도 있다 — 후자는 MuZero의 선택이고, §3.4의 가치 등가 원리 관점에서는 오히려 바람직하다.

식 (2)는 하나의 스칼라이고 한 번의 backward로 학습된다. 요구사항의 "단일 목적 함수"는 문법적으로 보존된다. 그러나 두 개의 sg가 식 (1)의 세 항을 다음과 같이 재배선한다.

비평가 경로에는 [C]형 항 $\big(V_\theta(h_t) - y_t^\lambda\big)\nabla_\theta V_\theta(h_t)$만 남는다. 타깃 안의 $r_t$와 $V(h_{t+1})$이 sg에 갇혔으므로 **[A](자기 파괴)와 [B](역방향 부트스트랩)는 대수적으로 소멸**한다. 행위자 경로에서 $\nabla_\theta r_t$는 계수 $-\beta\gamma^t < 0$의 **고정된 부호**로만 흐른다: 보상은 언제나 상승 방향, 즉 CE는 언제나 하강 방향으로만 최적화되며, 이 방향은 $\delta$의 부호와 무관한 구조적 사실이다. 지평선 너머의 미래는 상태를 경유해 흘러드는 $\nabla_h V_{\bar\theta}(h_H)$가 압축 전달한다(비평가 파라미터는 이 경로에서 동결 — DDPG에서 행위자가 $\nabla_a Q$만 빌려 쓰고 $Q$의 파라미터는 건드리지 않는 것과 동일한 비대칭).

### 2.3 정리 B — 연쇄 법칙 동치: 역전파는 정확한 결정론적 정책 그래디언트다

당신의 구조에서 동역학은 $g(h, u) = h + u$, $u = \mu_\theta(h) = f_\theta(h)$이고 $\partial g / \partial u = I$이다. 상태 기반 보상 $r(h)$에 대해 행동가치를 $Q^\mu(h, u) := r(h) + \gamma V^\mu(h + u)$로 정의하자.

**1-스텝의 경우 ($H=1$).** 행위자 항의 그래디언트는 연쇄 법칙에 의해

$$
\nabla_\theta \Big[ r(h) + \gamma V_{\bar\theta}\big(h + f_\theta(h)\big) \Big]
= \nabla_\theta f_\theta(h) \cdot \gamma \nabla_{h'} V_{\bar\theta}(h')
= \nabla_\theta \mu_\theta(h) \cdot \nabla_u \hat Q(h, u)\Big|_{u = \mu_\theta(h)}. \tag{3}
$$

식 (3)의 우변은 결정론적 정책 그래디언트 정리(Silver et al., 2014)가 규정하는 행위자 갱신 $\nabla_\theta J = \mathbb{E}_{h \sim \rho^\mu}\big[\nabla_\theta \mu_\theta(h)\, \nabla_u Q^\mu(h,u)|_{u=\mu(h)}\big]$ 그 자체다. 차이는 단 하나: DDPG가 학습된 비평가 $Q_\phi$로 근사해야 했던 $\nabla_u Q$를, 화이트박스 보상과 화이트박스 동역학의 **해석적 미분으로 대체**했다는 점이다. 이것이 정확히 SVG(1) (Heess et al., 2015)이다.

**일반 $H$-스텝의 경우.** $J = \sum_{k=0}^{H-1}\gamma^k r(h_k) + \gamma^H V_{\bar\theta}(h_H)$, $\lambda_t := \partial J / \partial h_t$로 정의하면 수반(adjoint) 재귀

$$
\lambda_H = \gamma^H \nabla V_{\bar\theta}(h_H), \qquad
\lambda_t = \gamma^t \nabla r(h_t) + \Big(I + \tfrac{\partial f}{\partial h}\Big|_{h_t}\Big)^{\!\top} \lambda_{t+1}, \qquad
\nabla_\theta J = \sum_{t} \Big(\tfrac{\partial f_\theta(h_t)}{\partial \theta}\Big)^{\!\top} \lambda_{t+1} \tag{4}
$$

이 성립하며, 이는 BPTT가 계산하는 양의 정의와 문자 그대로 일치한다. 식 (4)의 $\lambda$-재귀는 이산 Pontryagin 코스테이트 방정식이고(§3.2), 각 시점의 $\lambda_{t+1}$은 $n$-스텝 행동가치의 행동 그래디언트 $\nabla_u Q^{(H-t)}$를 정확히 담는다.

세 가지 함의를 명시한다. 첫째, **BPTT가 계산하는 $\nabla_\theta J$는 유한 지평 결정론적 정책 그래디언트의 근사가 아닌 참값**이다(DPG 정리는 이것의 정상분포 극한형으로, 방문 분포의 $\theta$-의존성을 무시하는 근사를 포함하지만 유한 지평 BPTT는 그 항까지 정확히 미분한다). 둘째, '절단 지평선 + 말단 가치 부트스트랩' 형태의 식 (2)는 미분가능 시뮬레이터 RL의 표준 정식화인 SHAC (Xu et al., 2022) 및 Dreamer의 상상 속 행위자 학습(Hafner et al., 2020–2023)과 동일 구조이며, 차이는 시뮬레이터가 외부 물리엔진이나 학습된 세계모델이 아니라 **네트워크 자신의 잔차 재귀**라는 점뿐이다. 셋째, 따라서 이 구조의 안정성은 새로 증명할 필요 없이 해당 계열의 이론과 대규모 경험적 검증을 그대로 상속한다.

### 2.4 정리 C — 네 병리의 대수적 소거

**P1 (브로드캐스팅).** 가치가 스칼라 헤드이므로 TD 회귀는 스칼라 대 스칼라다. $h$가 받는 그래디언트는 "미래 보상·가치를 개선하라"는 방향성분뿐이며, 랭크-1 복제 제약은 존재하지 않는다.

**P2 (자기 파괴).** $L_V$에 $\nabla_\theta r$ 항이 없다(sg 내부). 보상 그래디언트가 등장하는 유일한 장소는 $-\beta \sum \gamma^t \nabla_\theta r_t$이고 부호가 고정이므로, 어떤 $\delta$ 값에서도 CE를 올리는 갱신은 발생할 수 없다. $\blacksquare$

**P3-1 (지름길).** $v$가 $h$의 함수이므로 자유 슬랙이 아니며, 설령 비평가가 일시적으로 과소평가하더라도 행위자 신호 $\nabla_\theta J$는 $\delta$와 독립이므로 잔차를 0으로 만드는 것이 행위자에게 아무 보상이 되지 않는다. 과소평가는 sg 타깃(비평가가 움직일 수 없는 $r$을 포함)에 대한 회귀가 다음 스텝에 교정한다.

**P3-2 (동결).** $\delta \equiv 0$(완벽한 비평가)일 때 $\nabla_\theta J$는 오히려 **정확한 정책 그래디언트**가 되어 학습은 계속된다. 갱신이 멈추는 지점은 $\nabla_u Q = 0$, 즉 정책 개선 정리가 규정하는 수익의 진짜 정상점뿐이다. 탐험의 원천이 비평가의 '오차'가 아니라 비평가의 '기울기' $\nabla_h V$로 바뀌었기 때문이다. $\blacksquare$

**P4 (진동).** 비평가는 MSE 회귀이므로 갱신이 $\delta$에 비례하는 선형 수축 동역학이다(고정점: 사영된 $T^\pi v$). 행위자 스텝은 $\|\nabla \mathrm{CE}\|$에 비례하며, CE는 로짓에 대해 매끄러우므로 최적점 근방에서 자연 감쇠한다. L1의 상수 크기 스텝이 만들던 극한 주기는 발생 조건 자체가 사라진다. 꼬리가 두꺼운 타깃이 문제라면 Huber 손실이 원리적 대안이다(DQN의 선택과 동일한 근거: 이것은 회귀 문제이기 때문이다). $\blacksquare$

### 2.5 정리 D — 2-시간척도 수렴

비평가를 빠르게, 행위자를 느리게 둔다: $\alpha_V \gg \beta\alpha$, 또는 동등하게 EMA 계수 $\tau$로 타깃을 지연시킨다. Borkar의 2-시간척도 확률근사 이론에 의해, (i) 빠른 계는 준정적 정책 $\mu_\theta$ 하에서 비평가 ODE $\dot v = -(v - \Pi T^{\mu} v)$를 따르고, $T^\mu$는 $\gamma$-수축이므로 고정점 $v^\mu$로 수렴한다(선형 FA + on-policy에서는 Tsitsiklis & Van Roy 1997의 엄밀한 수렴 정리가 적용된다 — 당신의 롤아웃은 현재 $\theta$가 생성하므로 정의상 on-policy이고, deadly triad의 off-policy 다리가 애초에 없다). (ii) 느린 계는 $v \approx v^\mu$ 하에서 $\nabla J$를 따르는 경사 상승이 되어 수익의 국소 최적으로 수렴한다(Konda & Tsitsiklis, 2000의 액터-크리틱 수렴 구조). 비선형 함수근사에서 전역 보장이 없는 것은 이 분야 전체의 표준적 상태이며, EMA 타깃과 시간척도 분리가 그 간극을 메우는 검증된 장치다.

### 2.6 재귀 구조 특유의 추가 안정화 — 그리고 "왜 모든 것이 미분 가능한데도 비평가가 필요한가"

이 질문은 이 구조의 심장부에 있다. 전부 미분 가능하다면 $J = \sum_t \gamma^t r_t$를 무한 지평으로 직접 BPTT하면 되지 않는가? 안 된다. 재귀의 야코비안 곱 $\prod_t (I + \partial f/\partial h|_{h_t})$은 스펙트럼 반경이 1을 넘는 구간에서 지수적으로 폭발하며, 동역학이 카오스적일수록 그래디언트의 분산이 지평선에 대해 지수적으로 커진다는 것이 미분가능 시뮬레이션 분야의 확립된 병리다(Metz et al., 2021, "Gradients Are Not All You Need"; Parmas et al., 2018). **비평가는 이 지수적 분산을 유한한 함수근사 편향과 맞바꾸는 '시간의 압축기'다** — 지평선을 $H$에서 절단하고 그 너머를 $V$ 하나로 갈음하는 것. 이것이 SHAC와 Dreamer가 완전 미분 가능한 세계 안에서조차 비평가를 유지하는 이유이고, 당신의 구조에서 가치 채널이 장식이 아니라 필수인 이유다.

보조 장치 두 가지도 이론에 뿌리가 있다. 첫째, $f$에 스펙트럴 정규화를 걸어 $\|\partial f/\partial h\| \le 1 - \epsilon$을 강제하면 전이 $h \mapsto h + f(h)$의 재귀가 다루기 좋아질 뿐 아니라, $f$ 자체를 수축 조건 하에 두면 Banach 고정점 정리에 의해 재귀의 평형 존재·유일성이 보장된다 — 이는 Deep Equilibrium Model의 적정성(well-posedness) 조건과 정확히 같은 수학이다(Bai et al., 2019; Winston & Kolter, 2020). 둘째, 탐험이 필요하면 결정론적 정책에 재매개변수화 노이즈 $u_t = f_\theta(h_t) + \sigma_\theta(h_t)\varepsilon$을 더한다. 이는 SVG의 확률적 원형 그대로이며 미분 가능성을 보존한다.

---

## 3. 해석: 가치–상태 통합 흐름의 기하학 (요구사항 2)

### 3.1 재귀는 HJB 방정식의 특성곡선이다

$h_{t+1} = h_t + f(h_t)$는 스텝 1의 전진 오일러, 즉 연속 흐름 $\dot h = f(h)$의 이산화다(Neural ODE의 관점, Chen et al., 2018). 스텝폭 $\Delta t$와 $\gamma = e^{-\rho \Delta t}$를 넣고 극한을 취하면

$$
\frac{\delta_t}{\Delta t} \;=\; \frac{r\,\Delta t + (1 - \rho\Delta t)\, v(t{+}\Delta t) - v(t)}{\Delta t} \;\longrightarrow\; r + \dot v - \rho v,
$$

따라서 TD 오차의 기댓값을 0으로 만드는 것은 궤적 위에서

$$
\rho\, v(h) \;=\; r(h) \;+\; \nabla v(h) \cdot f(h) \tag{5}
$$

를 강제하는 것이다. 식 (5)는 정책 평가의 연속시간 Bellman 방정식이며, 제어 $u$에 대한 최대화를 붙이면 Hamilton–Jacobi–Bellman PDE가 된다(연속시간 TD: Doya, 2000). 여기서 결정적 통찰: **PDE를 상태공간 전체에서 푸는 대신, 시스템이 스스로 생성한 궤적 — 특성곡선 — 을 따라가며 콜로케이션으로 푸는 것이 TD 학습이고, 그 특성곡선을 생성하는 것이 당신의 재귀 루프다.** 증강 상태 $[v_t, h_t]$는 임의의 벡터 연결이 아니라, 가치함수 $V$의 그래프 $\{(h, V(h))\}$ 위의 한 점을 자기 흐름으로 수송(transport)하는 좌표다. 가치가 상태와 한 텐서에 실려 흐른다는 설계는 method of characteristics의 신경망 구현이다.

### 3.2 역전파는 Pontryagin 수반 방정식이다 — 혼합 표상의 진짜 이름은 위상공간

식 (4)의 $\lambda$-재귀를 다시 보라. 이는 최적제어의 이산 코스테이트(costate) 방정식이며, 역전파가 곧 수반법(adjoint method)이라는 고전적 동일성의 한 사례다. 최적성 이론의 핵심 항등식 — 최적 궤적 위에서 코스테이트는 가치함수의 기울기와 일치한다, $\lambda(t) = \nabla_x V(x(t))$ — 을 대입하면 구조의 역할 분담이 투명해진다. Pontryagin 최대원리는 본래 열린-루프의 2점 경계값 문제(TPBVP)다: 상태는 앞으로, 코스테이트는 뒤로 적분해야 하고 말단 조건이 필요하다. **비평가는 그 말단 코스테이트 $\lambda_H = \nabla V_{\bar\theta}(h_H)$를 폐루프 피드백으로 공급하는 장치**이고, 행위자의 BPTT는 해밀토니안 $\mathcal{H}(h, \lambda, u) = r(h) + \lambda^\top(h + f)$에 대한 $\nabla_u \mathcal{H}$ 상승, 즉 PMP의 최대화 조건을 경사법으로 실행하는 것이다.

그러므로 '가치–상태 혼합 표상'의 기하학적 정체는 이것이다: 순전파가 실어 나르는 것은 $(h, V(h))$ — 가치함수 그래프 위의 점이고, 역전파가 실어 나르는 것은 $(h, \nabla V(h))$ — **코탄젠트 번들의 단면, 즉 위상공간(phase space)의 점**이다. Hamilton–Jacobi 이론에서 $V$는 최적 흐름의 생성함수(generating function)이며 최적 동역학은 그 특성곡선이다. 가치는 상태에 대한 외부의 관전평이 아니라 흐름을 생성하는 기하적 대상 그 자체이므로, 둘을 하나의 텐서 흐름에 접어 넣은 당신의 설계는 우연한 편의가 아니라 **최적제어의 본래 기하와 동형**이다. 붕괴의 원인은 기하가 아니었다. PDE 잔차의 충족(평가)과 해밀토니안의 최대화(제어)라는, 위상공간의 서로 다른 두 방향의 연산을 하나의 스칼라 그래디언트에 묶은 배선이었다.

### 3.3 에너지 동역학과 리아푸노프 인증서로서의 가치

$E(h) := \mathrm{CE}(h) \ge 0$을 에너지로 정의하면, 잘 학습된 폐루프는 에너지를 낮추는 흐름이어야 한다. $W := -V$ (할인 누적 CE, 비음)로 두면 식 (5)로부터 $\dot W = \rho W - E$이고, 무할인 극한 $\rho \to 0$에서 $\dot W = -E \le 0$: **가치(의 음)는 학습된 추론 동역학의 리아푸노프 함수이며, 비평가 학습은 자기 자신의 재귀 루프에 대한 안정성 인증서(stability certificate)를 학습하는 행위다.** 이 관점은 실용적 선택지도 연다: $f = -\nabla_h \Phi_\theta$ (입력-볼록 신경망 ICNN 등으로 퍼텐셜을 학습)로 제약하면 추론이 문자 그대로 에너지 하강이 되어 재귀의 수렴이 구조적으로 보장된다 — Hopfield 및 예측부호화 계열과 접속하는 지점이다. DEQ 관점의 일관성 조건도 얻는다: 평형 $f(h^*) = 0$에서는 $V(h^*) = r(h^*)/(1-\gamma)$가 성립해야 하며, 이는 학습 중 공짜로 얻는 진단 지표다.

### 3.4 가치 등가 MDP, 자기참조 MDP, 그리고 정지 규칙

은닉 채널 $h$가 입력을 복원할 의무 없이 오직 보상·가치 예측에만 복무한다는 점이 불안하게 느껴질 수 있으나, 이는 정확히 **가치 등가 원리**(value equivalence, Grimm et al., 2020)가 정당화하는 바다: 모델은 관측에 대해서가 아니라 가치·보상에 대해서만 참 MDP와 등가이면 충분하며, 그런 표상이 오히려 계획에 최적이다. MuZero의 순환 잠재 동역학 + 보상·가치 헤드, Dreamer의 잠재 상상 속 BPTT 행위자 + sg TD($\lambda$) 비평가는 이 프로그램 전체의 대규모 존재 증명이다. 당신의 제안은 그 특수화다: **세계모델이 분류기 자신의 잔차 동역학이고 보상이 자신의 손실인 자기참조 MDP** $\mathcal{M}_\theta = (\mathcal{H},\, u = f_\theta,\, P: h \mapsto h+u,\, r = -\mathrm{CE})$. 환경·모델·에이전트의 경계가 소멸하는 대신 치르는 대가는 MDP 자체가 학습 중 비정상(nonstationary)이라는 것이고, 따라서 안정성은 환경에서 올 수 없으며 절차 — 그래디언트 라우팅과 시간척도 분리 — 에서 와야 한다. GAN과 자기대국(self-play)이 같은 처방을 필요로 하는 것과 정확히 같은 이유다.

마지막으로 가치 채널의 실용적 승격 하나. 스텝당 계산 비용 $\kappa$를 도입하면 "한 스텝 더 도는 것"은 그 자체로 최적 정지 문제가 되고, 재귀는 $\gamma V(h_{t+1}) - V(h_t) > \kappa$인 동안만 지속하는 것이 최적이다. 즉 가치 채널은 ACT/PonderNet류의 휴리스틱 정지 장치를 **원리적으로 대체하는 적응적 계산(adaptive computation) 기준**을 공짜로 제공한다.

---

## 4. 실행 레시피와 진단 지표

```text
초기화: 트렁크 f_θ, 헤드 V_θ, C_θ;  타깃 θ̄ ← θ
반복:
  h_0 ← encoder(입력);  h_{t+1} = h_t + f_θ(h_t),  t < H       # 하나의 재귀
  r_t = -CE(y, C_θ(h_t))                                        # 화이트박스 보상
  y_t^λ = TD(λ) 타깃 (r_t, V_θ̄(h_{t+1}) 사용, 전체 sg)
  L = Σ ½(V_θ(sg[h_t]) - sg[y_t^λ])²  -  β(Σ γ^t r_t + γ^H V_θ̄(h_H))
  θ ← θ - α ∇L        # 한 번의 backward; 라우팅은 sg가 수행
  θ̄ ← (1-τ)θ̄ + τθ    # Polyak;  τ 및 β로 2-시간척도 구현
```

권장 진단 지표: (i) $\mathrm{corr}(\delta_t, \Delta r_t) \approx 0$ — Phase 2 소거의 직접 검증(양의 상관이 재출현하면 라우팅 누수), (ii) $\big\|\prod_t (I + \partial f/\partial h)\big\|$의 로그 추적 — BPTT 폭주 감시, 필요시 스펙트럴 정규화, (iii) 평형 일관성 $|V(h^*) - r(h^*)/(1-\gamma)|$, (iv) 비평가 손실이 행위자 개선보다 빠르게 감소하는지 — 시간척도 위계의 확인.

## 5. 정직한 한계

비볼록 최적화이므로 국소 최적 수렴만 보장된다(분야 표준). 자기참조 MDP의 비정상성은 시간척도 분리로 관리되지만 제거되지는 않는다. 결정론적 정책은 탐험을 데이터 다양성 또는 재매개변수화 노이즈에 의존한다. 비선형 함수근사 하의 전역 수렴 이론은 미해결이며, 본 보고서의 보장은 (a) 선형 FA + on-policy에서의 엄밀 수렴, (b) 비선형에서의 국소 수렴 + EMA/시간척도에 의한 경험적 안정화라는, 현재 이론이 제공하는 최대치다.

## 참고문헌

Sutton (1988), *Learning to predict by the methods of temporal differences*. · Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed. (semi-gradient, GPI, deadly triad). · Baird (1995), *Residual algorithms*. · Tsitsiklis & Van Roy (1997), TD($\lambda$)의 선형 FA 수렴. · Konda & Tsitsiklis (2000), *Actor-critic algorithms*. · Borkar (2008), *Stochastic Approximation* (2-시간척도). · Silver et al. (2014), DPG. · Lillicrap et al. (2016), DDPG. · Heess et al. (2015), SVG. · Xu et al. (2022), SHAC. · Metz et al. (2021), *Gradients Are Not All You Need*. · Parmas et al. (2018), PIPPS. · Hafner et al. (2020–2023), Dreamer 계열. · Schrittwieser et al. (2020), MuZero. · Grimm et al. (2020), 가치 등가 원리. · Doya (2000), 연속시간·공간 RL. · Chen et al. (2018), Neural ODE. · Bai et al. (2019), DEQ; Winston & Kolter (2020), Monotone DEQ. · Pontryagin et al. (1962), 최적과정의 수학적 이론. · Amos et al. (2017), ICNN. · Graves (2016), ACT; Banino et al. (2021), PonderNet.

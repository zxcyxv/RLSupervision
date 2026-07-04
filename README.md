# RVSAC — Recursive Value-State Actor-Critic

`recursive-value-state-actor-critic-analysis.md`(이하 "분석 문서")의 식 (2) 아키텍처를
TRM(`ref/TinyRecursiveModels`)의 물리적 형태 위에 구현한 것.

## 아키텍처

순수 잔차 RNN 재귀 (TRM의 x/y/z 3변수, H/L 이중 주기 없음):

```
h_0     = embed(input tokens) + puzzle_emb        # encoder
u_t     = f_θ(h_t)                                # 전이 헤드 = 결정론적 정책 (Transformer 블록 스택)
h_{t+1} = h_t + u_t                               # t = 0..H-1
v_t     = V_θ(h_t)                                # 스칼라 가치 헤드
r_t     = -CE(y, C_θ(h_{t+1}))                    # 화이트박스 보상 (도착 상태 기준)
```

단일 손실, 단일 backward, sg 두 개로 미분 경로 3중 분리 (분석 문서 식 2):

```
L(θ) = Σ_t ½(V_θ(sg[h_t]) − sg[y_t^λ])²          # 비평가: TD(λ) 회귀
       − β(Σ_t γ^t r_t + γ^H V_θ̄(h_H))           # 행위자: 수익 BPTT (SHAC/SVG 동형)
θ̄ ← (1−τ)θ̄ + τθ                                  # Polyak 타깃 (2-시간척도)
```

## TRM에서 채택 / 폐기

| 채택 | 폐기 |
|---|---|
| Transformer 블록 (post-norm, Attention+SwiGLU) | x/y/z 3변수 carry 및 입력 재주입 |
| RoPE, puzzle embedding (+SignSGD sparse optim) | H_cycles/L_cycles 이중 주기 재귀 |
| stablemax CE, bfloat16, AdamATan2(→AdamW 폴백) | ACT Q-halting (q_head, halt 탐험) |
| 데이터 파이프라인 (puzzle_dataset, 빌더) | deep-supervision segment carry |
| EMA(평가용), hydra 설정 구조, 학습 루프 골격 | Bellman 잔차형 손실 일체 |

## 문서와 다르게 해석한 지점 (2건)

1. **보상을 도착 상태에 정의**: `r_t = -CE(C_θ(h_{t+1}))`. 레시피 원문은 `C_θ(h_t)`지만
   그대로면 마지막 상태 `h_H`의 분류기가 어떤 손실에도 나타나지 않아 평가가 불가능해진다.
2. **f 내부 입력 정규화**: `u = W_out(Blocks(rms_norm(h)))`, `W_out`은 zero-init(ReZero).
   외부 재귀는 순수 잔차로 유지하면서 h의 크기 표류만 차단한다.

평가기 호환: TRM ARC evaluator가 요구하는 `q_halt_logits` 자리에 `V̄(h_H)`(기대 수익)를
답 랭킹 점수로 노출한다.

## 파일

- `rvsac/model.py` — 트렁크/헤드/롤아웃, 타깃 가치 헤드(Polyak)
- `rvsac/losses.py` — 식 (2) 손실 + §4 진단 지표 (corr(δ,Δr), 평형 일관성, ‖u‖/‖h‖)
- `pretrain.py` — 학습 루프 (TRM 개작: ACT carry 제거, `update_target(τ)` 추가)
- `config/arch/rvsac.yaml` — γ=0.95, λ=0.9, β=0.1, H=16, τ=0.005
- `tests/test_routing.py` — 정리 C 검증 (그래디언트 라우팅 대수적 소거 확인)

## 옵션: K-step segmented BPTT

기본값 `arch.bptt_segment=0`은 기존 full-horizon BPTT다. `arch.bptt_segment=8`처럼 지정하면
forward rollout은 `arch.horizon`만큼 유지하되, 매 K step마다 `h`를 detach하고 segment별
K-step actor/critic bootstrap을 사용한다. 예:

```bash
DISABLE_COMPILE=1 WANDB_MODE=offline python pretrain.py \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" global_batch_size=128 \
  epochs=8000 eval_interval=2000 \
  arch.horizon=32 arch.bptt_segment=8 +run_name=h32-k8-b128
```

추가 지표:

- `bptt_segment`, `num_segments` — 현재 K와 segment 수
- `segment_ce_gain` — segment 내부 CE 개선량 평균 (`ce_start - ce_end`)
- `bootstrap_to_reward` — actor segment return에서 terminal value bootstrap이 reward 합 대비 차지하는 크기
- `segment_terminal_mean` — segment boundary의 `V̄(h_end)` 평균

## 실행

```bash
# 데이터 (예: Sudoku-Extreme 축소판)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-tiny --subsample-size 200 --num-aug 10

# 라우팅 테스트
python -m tests.test_routing

# 학습
DISABLE_COMPILE=1 WANDB_MODE=disabled python pretrain.py \
  data_paths="[data/sudoku-tiny]" global_batch_size=64 epochs=600 eval_interval=200 \
  arch.horizon=8 arch.hidden_size=256
```

## 진단 지표 읽는 법 (분석 문서 §4)

- `corr_delta_dr` — 롤아웃 내부 corr(δ_t, Δr_t). 주의: 문서 §4의 지표는 "업데이트가 유발한
  보상 변화"와의 상관(학습 스텝 간)이고, 이 지표는 롤아웃 내 자연 상관이라 양수가 정상.
  Phase 2 소거의 엄밀한 검증은 `tests/test_routing.py`의 dL/dCE = +βγ^t 항등식 테스트가 수행
- `ce_last < ce_first` — 재귀 흐름이 답을 실제로 개선하는지
- `u_h_ratio` — BPTT 폭주 감시; 지속 상승 시 스펙트럴 정규화 고려
- `eq_gap` = |V(h_H) − r_H/(1−γ)| — 평형 일관성 (수렴기에만 유의미)
- `td_abs` — 비평가가 행위자보다 빨리 수렴하는지 (시간척도 위계)

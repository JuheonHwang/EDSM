# 원본 코드 대비 달라진 점

**한국어** | [English](legacy_changes.en.md)

이 문서는 원본 스크립트 `EDRegisterSparse_Show_iter_*.m` 8개를 써 본 사람을 위한 것입니다. 새 구조(`+edsm/`)가 무엇을 같게 하고 무엇을 바꿨는지, 바뀐 것은 어떻게 검증했는지, 원본의 숫자를 다시 얻으려면 어떻게 하는지를 정리합니다. 원본 스크립트는 작업 트리에서 지워졌고 git 이력에만 남아 있습니다.

## 1. 원본 스크립트와 새 호출의 대응

모든 새 호출은 먼저 모델과 시작점을 만듭니다.

```matlab
cfg   = edsm.defaultConfig();
src   = edsm.loadMesh('data/simple_poisson_mesh_020.ply');
tar   = edsm.loadMesh('data/simple_poisson_mesh_022.ply');
model = edsm.buildModel(src, tar, cfg);
P0    = edsm.initParam(model);
```

| 원본 스크립트 | 새 호출 | 비고 |
|---|---|---|
| `..._gradient_hjh.m` | `edsm.solvers.gradientDescent(P0, model, struct('maxIter', 100, 'stepSize', 1e-3, 'globalStepScale', 0.01))` | 원본은 오래된 Jacobian 변형을 씁니다(3절 D). 수치 재현은 시작점의 `err`만 가능합니다. |
| `..._Gauss_Newton_hjh_re.m` | `edsm.solvers.gaussNewton(P0, model, struct('maxIter', 100, 'stepTol', 1e-3/(1-1e-3)))` | `stepTol`은 원본 종료식과 같게 맞춘 값입니다(3절 J). |
| `..._Gauss_Newton_delta_hjh.m` | `edsm.solvers.gaussNewton(P0, model, struct('jacobian', 'numeric', 'rankTol', 1e-5))` | 유한 차분 Jacobian. 원본은 항 가중치를 두 번 적용했습니다(3절 E). 재현하지 않습니다. |
| `..._LM_hjh.m` | (이식하지 않음) | `LM_hjh_2`에 대체되었습니다(3절 F). |
| `..._LM_hjh_2.m` | `edsm.solvers.levenbergMarquardt(P0, model)` | 기본값이 `damping 'diag'`, `linearSolver 'direct'`, `lambda0 0.01`입니다. 원본에는 종료 기준이 없었으므로 재현 시 `gradTol = 0, funTol = 0`을 줍니다(4절). |
| `..._CG_hjh.m` | `edsm.solvers.levenbergMarquardt(P0, model, struct('linearSolver', 'cg'))` | `damping`은 `'diag'`. 원본의 감쇠 고정 버그를 고쳤습니다(3절 B). |
| `..._PCG_hjh.m` | `edsm.solvers.levenbergMarquardt(P0, model, struct('linearSolver', 'pcg'))` | 원본의 `mode`(`'none'`, `'eye'`, `'diag'`)는 옵션 `damping`입니다. 전처리기는 Jacobi(`diag(A)`)입니다. |
| `..._Jall_FINAL.m` | `edsm.solvers.lsqnonlinBaseline(P0, model)` | Optimization Toolbox 필요. 원본과 같은 옵션(해석적 Jacobian, `FunctionTolerance 1e-10`, `MaxIterations 30`)입니다. |

옵션 이름과 기본값은 각 함수의 도움말 머리말(`help edsm.solvers.levenbergMarquardt` 등)이 정답입니다. 솔버는 모두 `[P, hist] = solver(P0, model, opts)` 형태이며 `opts`에 모르는 필드를 주면 `edsm:solvers:unknownOption` 오류가 납니다(오타 방지).

원본이 스크립트 안에 하드코딩하던 설정은 `edsm.defaultConfig()`의 기본값과 같습니다(`nVertexNeighbor 4`, `nNodeNeighbor 6`, `sampleRate 0.01`, `wRot 1.0`, `wSmooth 0.1`, `wLap 0.01`, `distanceOnMesh true`, `seed 1`). 이 값을 바꾸려면 `cfg`의 필드를 바꿔 `edsm.buildModel`에 넘깁니다.

## 2. 의존성 제거

| 원본이 필요로 하던 것 | 새 코드 |
|---|---|
| mexopencv (`cv.Rodrigues`) | `edsm.rodrigues` (순수 MATLAB, Jacobian 포함) |
| Statistics Toolbox (`knnsearch`) | `edsm.nearestIdx` (정확한 완전 탐색, 동률은 가장 작은 행 번호) |
| GPU (`gpuArray`) | 사용하지 않음 |
| 비공개 패키지 `jwutils` (`plyread` 등) | `edsm.loadMesh`, `edsm.showMesh`, `edsm.showMeshes` |

`+edsm` 핵심은 기본 MATLAB R2020b 이상만 필요합니다. `lsqnonlinBaseline`은 Optimization Toolbox가 필요하고, `numericJacobian`의 `parfor`는 선택 사항입니다. 개발과 테스트는 R2026b(Apple Silicon)에서만 했으며 이전 릴리스는 시험하지 않았으므로 호환성을 주장하지 않습니다. 튜토리얼 스크립트의 백분위수 계산도 Statistics Toolbox 없이 동작하도록 도우미 `tutorials/helpers/pct.m`을 씁니다.

## 3. 동작 차이와 수정한 버그

### A. 해석적 Jacobian의 세 가지 오류 (수정됨)

원본의 Jacobian에는 세 가지 오류가 있었습니다. 기본값(`cfg.legacyJacobian = false`)은 수정된 Jacobian이고, `cfg.legacyJacobian = true`는 원본을 비트 단위로 재현합니다. $F$는 두 모드에서 같습니다.

1. `Jrot1`/`Jrot2`가 행 우선(row-major) $A_i$ 기준으로 배치되어 있는데 $P$는 $A(:)$를 열 우선으로 저장합니다. $A_i$가 대칭일 때(예: 항등 시작점)만 맞습니다.
2. Laplacian 블록의 회전($r$) 열이 0으로 남아 있었습니다.
3. 유효한 ED 이웃이 없는 소스 정점 9개(노드가 없는 작은 분리 성분)는 $F$에서 소스 위치에 고정되는데 $J$는 변형된 것으로 취급했습니다.

**영향.** 원본 Jacobian으로는 LM이 `err` 162.4137 근처에서 멈추고(`lambdaMax`, 33번 반복) 그 점의 기울기 노름(2-노름)이 수정된 Jacobian으로 재면 42.46, 원본 Jacobian으로 재면 42.33입니다. 수정된 Jacobian은 150.6621에 도달하고 기울기 노름이 0.2019입니다([06장](06_levenberg_marquardt.md) 4절, `tutorials/output/t06_levenberg_marquardt.log`).

**검증.**
- [03장](03_jacobian.md)(`tutorials/t03_jacobian.m`)이 임의의 한 점에서 해석적 Jacobian을 유한 차분(`edsm.numericJacobian`, 대응 고정)과 블록별로 비교합니다. 수정된 Jacobian은 모든 블록의 최대 절대 오차가 5.42e-09 이하이고, 원본 Jacobian은 최대 9.64(적합 항 $r$ 열)입니다(`tutorials/output/t03_jacobian.log`).
- `tests/test_core.m`은 수정된 Jacobian을 모든 행과 열에서 단언하고, 원본 Jacobian은 출력만 합니다. 또한 `F`가 `legacyJacobian`과 무관한지 단언합니다.
- 원본 모드의 비트 재현은 `tests/test_regression.m`이 확인합니다(아래 4절).

### B. CG/PCG에서 감쇠가 고정되는 버그 (수정됨)

원본 CG/PCG 스크립트는 감쇠 행렬 $A=J^\top J+\lambda D$를 재시도 루프 **앞에서 한 번만** 만들었습니다. 한 단계가 거절된 뒤 $\lambda$를 10배 키워도 시스템이 바뀌지 않아 같은 해가 반복해서 거절되었고, 이 때문에 원본은 각각 4번(CG), 3번(PCG) 반복하고 멈췄습니다. `levenbergMarquardt`는 재시도마다 현재 $\lambda$로 행렬을 다시 만듭니다.

**검증.** `tests/test_regression.m`은 새 솔버가 원본이 멈춘 반복의 다음 반복(CG는 5번째, PCG는 4번째)에서 거절을 겪고도 계속 진행함을 단언합니다(`h.nReject(nBL + 1) > 0`).

### C. 첫 반복의 시험점 계산이 무효 정점을 변형함 (재현하지 않음)

원본 `LM_hjh_2`, `CG`, `PCG`의 `EfuncLocal`은 첫 바깥 반복에서 지역 복사본의 `idxNeighbor(NaN)`을 1로 덮어쓴 다음 시험점을 계산하려고 자기 자신을 호출합니다. 그래서 첫 반복의 시험점(기록된 첫 오차, 수락 판정, 2번째 반복이 쓰는 이전 오차)에서는 무효 정점 9개가 변형됩니다. 새 솔버는 올바른 $F$(무효 정점은 소스 위치)를 씁니다. 이 때문에 원본의 첫 반복 오차와 새 솔버의 값은 약간 다릅니다.

**검증.** `tests/test_regression.m`은 무효 정점의 이웃 인덱스를 1로 바꾼 모델로 첫 반복점의 $F$를 평가해 원본이 기록한 첫 오차와 일치함을 단언합니다(`lm_v2`는 상대 편차 `< 1e-12`, `cg`/`pcg`는 `< 1e-6`).

### D. 경사하강법 스크립트의 오래된 Jacobian 변형 (재현하지 않음)

원본 `gradient_hjh`에는 다른 일곱 스크립트보다 오래된 Jacobian(회전 열이 치환되지 않았고 $R$ 블록의 행 인덱스도 다름)이 들어 있습니다. `legacyJacobian = true`는 일곱 스크립트의 Jacobian을 재현하므로 경사하강법 스크립트의 반복 궤적은 재현되지 않습니다. 반복 전의 오차(시작점)만 일치합니다.

### E. 유한 차분 GN 스크립트가 항 가중치를 두 번 적용함 (재현하지 않음)

원본 `Gauss_Newton_delta_hjh`는 유한 차분 Jacobian에 가중치를 두 번 곱했습니다. 새 `numericJacobian`은 `edsm.residual`을 그대로 차분하므로 가중치가 한 번만 들어갑니다. 원본의 수치는 재현하지 않습니다. 새 코드에서 유한 차분 Jacobian을 쓰려면 잡음 수준의 특이값을 잘라내는 `rankTol`(약 `1e-5`)을 함께 줍니다. 유한 차분 Jacobian의 정확히 영인 특이값이 `1e-8` 정도의 잡음을 가져 기본 허용치로는 잘리지 않기 때문입니다(`gaussNewton` 도움말).

### F. 원본 LM v1은 이식하지 않음, `inv()` 제거

원본 `LM_hjh`(v1)는 `LM_hjh_2`로 대체되어 이식하지 않았습니다. 원본 LM 계열이 쓰던 `inv(...) * ...`는 `A \ b`로 바꿨습니다. 이 때문에 오차가 상대 `1e-6` 수준에서 달라질 수 있습니다(`tests/test_regression.m`의 `lm_v2` 비교 허용치가 `2e-6`입니다).

### G. `distanceOnMesh = false` 분기 (수정됨)

원본에서는 `Lap`이 측지 거리 분기 안에서만 정의되어 유클리드 거리 분기가 처음 사용하는 곳에서 오류로 중단되었습니다. `edsm.buildModel`은 두 분기에서 같은 식으로 `Lap`을 만듭니다. `tests/test_core.m`이 이 분기를 실행합니다.

### H. 회전 벡터 규약

원본 스크립트의 `encodeParam`은 MATLAB의 `rotationMatrixToVector`를 썼는데, 그 규약은 `cv.Rodrigues`와 전치 관계여서 $R=I$일 때만 일치합니다. 새 `edsm.encodeParam`은 `edsm.rodrigues`와 일관된 역변환을 씁니다. 항등 시작점에서는 차이가 없습니다.

### I. 새로 생긴 종료 기준 (동작 변경)

원본에는 LM에 수렴 판정이 없었습니다. 새 `levenbergMarquardt`는 기본으로 `gradTol = 1e-6`(시작점 기울기의 무한 노름 대비 상대값)과 `funTol = 1e-8`(3번 연속 작은 감소)을 씁니다. 그래서 `maxIter = 100`을 주고도 일찍 끝날 수 있습니다(예: 기본 설정은 48번 반복에서 `funTol`로 종료, 06장). 이전 동작은 `gradTol = 0, funTol = 0`입니다.

### J. Gauss–Newton의 종료 기준 해석

원본 종료식은 첫 두 항을 벡터 $F$에 대해 원소별로 비교하므로 모든 성분이 만족할 때만 발동합니다. 세 번째 항은 $\|\delta\|_\infty<\epsilon_2/(1-\epsilon_2)=1.001001\times10^{-3}$과 같습니다. 새 `gaussNewton`은 이 세 번째 항만 남기고 `stepTol`로 둡니다. 기본값은 `1e-3`이고 원본과 정확히 같게 하려면 `1e-3/(1-1e-3)`을 줍니다.

### K. 기타

- `edsm.buildModel`은 노드 선택 전후로 전역 난수 상태를 저장·복원합니다. 원본 스크립트는 `rng(1,'twister')`를 직접 호출해 전역 상태를 바꿨습니다.
- `edsm.loadMesh`는 `jwutils.plyread`와 같은 $V$, $F$를 돌려줍니다(`tests/test_core.m` (d)는 비교용 원본 패키지가 없으면 건너뜁니다).
- `buildModel`의 `assert(~any(isnan(distNodeNeighbor(:))))`는 연결되지 않은 노드(거리 `Inf`)를 잡지 못하므로 보호 장치로 쓸모가 없습니다(코드에 같은 주석이 있습니다).
- 목적함수 $F$ 자체의 정의(에너지 항, 정규화, 가중치)는 같습니다. [02장](02_energy.md)이 이 정의를 설명합니다.

## 4. 원본의 숫자를 다시 얻으려면

1. **원본 Jacobian으로.** `cfg = edsm.defaultConfig(); cfg.legacyJacobian = true;`로 모델을 만듭니다.
2. **종료 기준 끄기.** LM에는 `struct('gradTol', 0, 'funTol', 0)`을 줍니다. 원본 LM은 `lambda > 1e5`(`lambdaMax`)이거나 반복 횟수(100)에 이를 때만 멈췄습니다. GN에는 `stepTol = 1e-3/(1-1e-3)`을 줍니다.
3. **회귀 테스트 실행.** `tests/test_regression.m`이 위 설정으로 새 솔버를 돌려 `tests/baseline/*.mat`(원본 스크립트의 수치 기록, 만든 방법은 `tests/baseline/README.md`)와 비교합니다.

```matlab
setup
run('tests/test_regression.m')
```

테스트가 단언하는 허용치는 다음과 같습니다(코드의 값).

| 대상 | 단언 |
|---|---|
| `gradient` | 시작점 오차가 `1e-14` 이내로 일치. 이후는 3절 D 때문에 달라지며 출력만 합니다. |
| `gauss_newton` | 반복 횟수와 종료 이유(`stepTol`)가 원본과 같고, 처음 18번 반복의 상대 편차가 `1e-10` 미만. 그 이후는 발산하는 혼돈 궤적이라 비교하지 않습니다. |
| `lm_v2` | $\lambda$ 수열(사용값과 다음 값)이 원본과 완전히 같고, 종료 이유가 `lambdaMax`. 2번째 반복부터 오차의 상대 편차가 `2e-6` 미만, 최종 $P$의 상대 편차가 `5e-4` 미만. 첫 반복은 3절 C 때문에 별도로 설명. |
| `cg`, `pcg` | 원본이 멈추기 전까지(CG 4번, PCG 3번)의 오차가 상대 `1e-3` 이내. 새 솔버는 그다음 반복에서 거절을 겪고도 계속합니다. |
| `lsqnonlin` | 처음 3번이 `1e-12`, 전체가 `1e-4` 미만. 새 호출은 31번 반복하고 `hist.err`는 시작점을 포함해 원본 기록과 같은 개수입니다. |

실행하면 케이스별 최대 상대 편차 표가 출력됩니다. 이 문서는 그 출력 값을 직접 인용하지 않고 코드의 단언 허용치만 인용합니다.

## 5. 의도적으로 재현하지 않는 것

- 원본 Jacobian의 비트 재현은 `legacyJacobian = true`로 하지만, 아래는 재현하지 않습니다.
  - 경사하강법 스크립트의 오래된 Jacobian 변형(3절 D)
  - 첫 반복의 시험점에서 무효 정점이 변형되는 효과(3절 C)
  - CG/PCG의 감쇠 고정으로 인한 조기 종료(3절 B)
  - 유한 차분 GN의 가중치 이중 적용(3절 E)
  - LM v1(`LM_hjh`)(3절 F)
- 원본의 그림 표시와 `pause`, 결과 저장(`jwutils.saveMesh` 등)은 이식하지 않았습니다. 솔버는 그림을 그리지 않고 `opts.callback`과 `opts.verbose`로 반복 정보를 내보냅니다. 시각화는 `edsm.showMesh`, `edsm.showMeshes`와 `tutorials/`의 도우미가 담당합니다.

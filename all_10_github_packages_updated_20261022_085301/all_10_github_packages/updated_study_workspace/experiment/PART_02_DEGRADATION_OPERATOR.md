# 02_degradation_operator: 확산–반응 열화 연산자

초안 대응: **§3.2**

주 구현: `docrestore/operators.py`.

보조 모듈: 없음.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/02_degradation_operator.txt
```

## 입력

[0,1] grayscale 영상, kappa, alpha, T, n_steps, h.

## 실행

```bash
python -m examples.02_operator_usage
```

## 출력과 구현 범위

NumPy 열화 결과와 PyTorch 미분 가능한 결과.

잉크 밀도 v=1-u에 확산–반응식을 적용합니다. DCT-II를 이용해 zero-flux 공간 이산화의 implicit Euler 해를 계산하며, 연산자 내부에서 양자화하거나 clip하지 않습니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/02_degradation_operator.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

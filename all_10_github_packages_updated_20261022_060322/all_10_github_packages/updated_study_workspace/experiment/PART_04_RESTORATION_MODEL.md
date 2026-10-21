# 04_restoration_model: 공통 U-Net 모델

초안 대응: **§4.1**

주 구현: `docrestore/model.py`.

보조 모듈: 없음.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/04_restoration_model.txt
```

## 입력

(B,1,H,W) float tensor. 모델 입력은 관측 영상 y입니다.

## 실행

```bash
python -m examples.04_model_usage
```

## 출력과 구현 범위

입력과 같은 크기의 [0,1] 복원 영상 tensor.

4-level U-Net, 기본 채널 32/64/128/256, GroupNorm/ReLU, nearest upsampling, sigmoid 출력입니다. 학습 orchestration은 07번에 들어 있습니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/04_restoration_model.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

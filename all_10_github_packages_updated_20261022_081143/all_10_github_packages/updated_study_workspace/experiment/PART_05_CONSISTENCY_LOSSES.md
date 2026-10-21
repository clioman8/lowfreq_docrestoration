# 05_consistency_losses: A/B/C 공통·일관성 손실

초안 대응: **§4.2–4.5**

주 구현: `docrestore/losses.py`.

보조 모듈: 없음.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/05_consistency_losses.txt
```

## 입력

복원·정답 영상, forward residual r=D_eta(pred)-y, 고정 Fourier 기저 Q, beta와 lambda.

## 실행

```bash
python -m examples.05_losses_usage
```

## 출력과 구현 범위

L1+SSIM 데이터 손실, full residual 에너지, 저주파 투영 residual 에너지.

A=data, B=data+lambda*full, C=data+lambda*subspace입니다. full과 subspace 모두 pixel 수 N으로 정규화합니다. 이 묶음은 이미 계산된 residual을 받으며 forward 연산자는 02번에 있습니다. 기저의 DC 포함·정렬·절단 관례는 함수 docstring에 명시했습니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/05_consistency_losses.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

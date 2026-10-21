# 07_training_and_plan: 학습·lambda 탐색·체크포인트

초안 대응: **§4.7·§5.3–5.4**

주 구현: `docrestore/train.py`, `docrestore/plan.py`.

보조 모듈: `common.py`, `model.py`, `operators.py`, `losses.py`, `assets.py`, `evaluate.py`, `metrics.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/07_training_and_plan.txt
```

## 입력

학습 clean manifest, validation S0–S4 데이터, config, Tesseract와 고정 영어 모델.

## 실행

```bash
python -m docrestore.plan --config configs/pilot.example.json --data-root data/pilot --tessdata-dir assets/tessdata_best --runs-root runs/pilot --output runs/pilot_plan.json
python -m docrestore.train fit --data-root data/pilot --config configs/pilot.example.json --group A --seed 0 --run-dir runs/pilot_A_seed0 --tessdata-dir assets/tessdata_best
```

## 출력과 구현 범위

argv 배열의 실행 계획, best.pt/last.pt, 학습 로그, 설정·seed provenance, best validation 이미지·OCR.

B/C는 --group B 또는 C와 --lambda-value를 지정합니다. plan은 계획만 작성하며 학습을 자동 시작하지 않습니다. 대응 seed, 동일 탐색 예산, validation 최대 shifted CER→평균 CER 순으로 선택합니다. 기본 lambda 3개·seed 3개 설계는 seed-0 선택 run 재사용 시 13회 실행/최종 9회 비교입니다. 최종 설정은 validation 후 동결합니다. resume은 구현하지 않았습니다.

## 검증

python -m unittest discover -s tests -v

분리된 파일의 검사 결과: `validation/07_training_and_plan.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

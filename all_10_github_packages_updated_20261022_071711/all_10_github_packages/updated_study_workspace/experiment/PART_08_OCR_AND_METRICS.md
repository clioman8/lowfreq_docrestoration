# 08_ocr_and_metrics: 고정 OCR·CER·영상 지표

초안 대응: **§5.5–5.6**

주 구현: `docrestore/evaluate.py`, `docrestore/metrics.py`.

보조 모듈: 없음.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/08_ocr_and_metrics.txt
```

## 입력

평가할 split만 포함한 manifest와 복원/참조 PNG, 고정 Tesseract와 eng.traineddata.

## 실행

```bash
python -m docrestore.evaluate --manifest runs/final_test/C_seed0/predictions_manifest.jsonl --images-root data/final --output runs/final_test/C_seed0_ocr.jsonl --mode restored --tessdata-dir assets/tessdata_best --method C --seed 0
```

## 출력과 구현 범위

표본별 OCR 텍스트·edit counts·CER·PSNR·SSIM·시간·OCR/모델 provenance JSONL.

evaluate는 manifest의 모든 행을 평가합니다. 최종 test만 담은 manifest를 넘깁니다. 학습 모델의 predict 명령은 07번입니다. CER는 NFC+공백 정리, 대소문자·구두점 유지, corpus edit distance/GT 길이이며 1로 자르지 않습니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/08_ocr_and_metrics.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

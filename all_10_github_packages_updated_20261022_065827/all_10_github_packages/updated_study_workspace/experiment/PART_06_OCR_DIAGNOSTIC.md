# 06_ocr_diagnostic: 학습 전 OCR 사전 검사

초안 대응: **§5.2**

주 구현: `docrestore/diagnose.py`.

보조 모듈: `common.py`, `evaluate.py`, `metrics.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/06_ocr_diagnostic.txt
```

## 입력

01·03번에서 만든 데이터, 별도 설치 Tesseract 엔진, 고정 eng.traineddata.

## 실행

```bash
python -m docrestore.diagnose --data-root data/pilot --config configs/pilot.example.json --tessdata-dir assets/tessdata_best --output-dir runs/pilot_diagnostic
```

## 출력과 구현 범위

validation clean/degraded OCR 기록, CER와 OCR 실행시간을 담은 진단 결과.

test는 사전 검사에서 읽지 않습니다. clean 가독성과 강도별 OCR 난도, 실행시간을 확인해 본학습 전에 설정을 고릅니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/06_ocr_diagnostic.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

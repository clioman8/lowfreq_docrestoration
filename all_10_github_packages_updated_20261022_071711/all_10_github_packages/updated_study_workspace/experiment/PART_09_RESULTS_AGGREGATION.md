# 09_results_aggregation: severity·seed 결과 집계

초안 대응: **§5.7–5.9·§6**

주 구현: `docrestore/aggregate.py`.

보조 모듈: `evaluate.py`, `metrics.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/09_results_aggregation.txt
```

## 입력

08번의 A/B/C 전 seed OCR JSONL과 clean/degraded 참조, 동결된 config.

## 실행

```bash
python -m docrestore.aggregate --results runs/final_test/A_seed0_ocr.jsonl runs/final_test/B_seed0_ocr.jsonl runs/final_test/C_seed0_ocr.jsonl --config configs/final.template.json --output-dir runs/final_tables --plot
```

## 출력과 구현 범위

severity별 CER/영상 지표, seed별 최대 CER와 평균·표본 SD, threshold crossing, CSV/JSON, 선택적 plot.

명령은 seed 0 경로만 보여주는 문법 예시입니다. 최종 비교에는 실제 전체 seed 경로를 나열합니다. severity별 corpus CER→seed별 최대→seed 평균·SD 순이며, 불완전한 조건을 완전한 주 결과로 보고하지 않습니다. 이 단계는 OCR 엔진을 실행하지 않습니다.

## 검증

이 묶음은 분리 후 import/CLI/API 예제 검사를 기록하며, 기존 전체 수치 테스트는 학습 패키지와 baseline 기록을 참고합니다.

분리된 파일의 검사 결과: `validation/09_results_aggregation.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

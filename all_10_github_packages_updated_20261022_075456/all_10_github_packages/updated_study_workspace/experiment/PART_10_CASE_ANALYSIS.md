# 10_case_analysis: 대표·심각·C 악화 사례 선정

초안 대응: **§5.10**

주 구현: `docrestore/cases.py`.

보조 모듈: `common.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/10_case_analysis.txt
```

## 입력

A/B/C 및 degraded의 같은 test 문서·severity 목록, 사전에 정한 example_seed와 threshold_adjacent_interval.

## 실행

```bash
python -m docrestore.cases --A runs/final_test/A_seed0_ocr.jsonl --B runs/final_test/B_seed0_ocr.jsonl --C runs/final_test/C_seed0_ocr.jsonl --degraded runs/final_test/degraded_ocr.jsonl --config configs/final.template.json --output runs/final_tables/cases.json
```

## 출력과 구현 범위

대표 입력, threshold 인접, 심각 입력, C-A 최대 악화 사례의 ID와 수치를 담은 cases.json.

선정 규칙은 결정적입니다. 해당하는 사례가 없으면 없음으로 기록합니다. 논문용 그림 배치와 사례 해석은 이 코드의 출력 범위에 포함하지 않습니다.

## 검증

python -m unittest discover -s tests -v

분리된 파일의 검사 결과: `validation/10_case_analysis.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

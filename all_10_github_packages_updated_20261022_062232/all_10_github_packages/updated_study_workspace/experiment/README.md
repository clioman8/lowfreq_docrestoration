# Document restoration: GitHub 업로드용 10개 기능 패키지

각 ZIP은 담당 구현과 실행에 필요한 로컬 의존 모듈을 포함합니다. 입력 데이터와 이전 단계의 산출물은 해당 단계에서 준비합니다. 현재 압축에 포함된 `PART_*.md`를 먼저 읽으십시오.

| 번호 | 초안 부분 | 담당 기능 | 개별 실행 안내 |
|---|---|---|---|
| 01 | §3.4·§5.1 | 데이터 원천·분할·렌더링 | `PART_01_DATA_PREPARATION.md` |
| 02 | §3.2 | 확산–반응 열화 연산자 | `PART_02_DEGRADATION_OPERATOR.md` |
| 03 | §3.3 | 동일 문서 S0–S4 생성 | `PART_03_SEVERITY_GENERATION.md` |
| 04 | §4.1 | 공통 U-Net 모델 | `PART_04_RESTORATION_MODEL.md` |
| 05 | §4.2–4.5 | A/B/C 공통·일관성 손실 | `PART_05_CONSISTENCY_LOSSES.md` |
| 06 | §5.2 | 학습 전 OCR 사전 검사 | `PART_06_OCR_DIAGNOSTIC.md` |
| 07 | §4.7·§5.3–5.4 | 학습·lambda 탐색·체크포인트 | `PART_07_TRAINING_AND_PLAN.md` |
| 08 | §5.5–5.6 | 고정 OCR·CER·영상 지표 | `PART_08_OCR_AND_METRICS.md` |
| 09 | §5.7–5.9·§6 | severity·seed 결과 집계 | `PART_09_RESULTS_AGGREGATION.md` |
| 10 | §5.10 | 대표·심각·C 악화 사례 선정 | `PART_10_CASE_ANALYSIS.md` |

## 설치와 실행

Python 3.10 이상을 사용합니다. 프로젝트 루트에서 다음을 실행합니다. `NN_name`은 현재 패키지 파일명으로 바꿉니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 / macOS·Linux: source .venv/bin/activate
python -m pip install -r requirements/NN_name.txt
```

개별 안내의 `python -m docrestore...` 명령을 이 폴더에서 실행합니다. 모델·연산자·손실은 CLI 대신 `python -m examples.02_operator_usage` 같은 API 예제를 제공합니다. 전체 패키지를 합쳤을 때는 `python -m pip install -r requirements.txt`로 공통 환경을 설치할 수 있습니다. Tesseract 엔진과 고정 영어 모델은 별도 준비합니다.

## 하나의 GitHub 저장소에 합치기

10개 ZIP을 같은 위치에 풀면 동일한 최상위 `lowfreq_docrestoration/` 안에 합쳐집니다. 같은 경로의 중복 파일은 내용이 동일하므로 덮어써도 됩니다. 데이터 생성과 severity 단계는 같은 `data.py`의 서로 다른 사용 경로입니다. 겹치는 코드를 따로 수정하기 시작하면 이후에는 직접 변경을 병합해야 합니다.

GitHub에는 ZIP을 푼 뒤 `lowfreq_docrestoration/` **안의 파일과 폴더**를 올립니다. `docrestore/`, `configs/`, `requirements/`, `README.md`, `pyproject.toml` 등의 상대 경로를 유지하십시오. 각 ZIP만 풀어 담당 단계의 별도 저장소로 사용하는 것도 가능합니다.

원문 archive, 폰트 파일, OCR traineddata, 생성 데이터, 학습 checkpoint는 이 소스 묶음에 포함하지 않았습니다. 배치 경로와 공식 원천은 `SOURCES.md`, 전체 실험 흐름은 `docs/FULL_EXPERIMENT_GUIDE.md`에 있습니다.

## 검증 범위

`packaging/NN_name.json`에 담당 파일·보조 모듈·원본 SHA-256이 있습니다. `validation/NN_name.json`에는 각 ZIP을 별도 폴더에 풀어 확인한 import·CLI·API 예제 결과가 있습니다. 포함된 기존 테스트는 `python -m unittest discover -s tests -v`로 실행합니다. 테스트가 없는 일부 패키지는 import/CLI/API 기능검사만 기록합니다.

`validation/starter_baseline.json`은 분리 전 전체 starter의 기존 검사 기록입니다. GPU 본학습, 실제 전체 SLR11 처리, Tesseract 5.5.3 실행, 최종 성능 비교는 아직 검증하지 않았습니다. 예제 severity·lambda·학습 예산은 검증된 논문 설정이 아닙니다.

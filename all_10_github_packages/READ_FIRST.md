# GitHub 업로드용 10개 ZIP

10개 ZIP을 같은 폴더에 풀어 `lowfreq_docrestoration/` 안의 소스를 GitHub에 올립니다. 각 ZIP은 담당 파일과 필요한 보조 코드, 설치·실행 안내를 포함합니다. 외부 원문·폰트·OCR 모델은 별도 준비합니다.

- `01_data_preparation.zip`: §3.4·§5.1 데이터 원천·분할·렌더링
- `02_degradation_operator.zip`: §3.2 확산–반응 열화 연산자
- `03_severity_generation.zip`: §3.3 동일 문서 S0–S4 생성
- `04_restoration_model.zip`: §4.1 공통 U-Net 모델
- `05_consistency_losses.zip`: §4.2–4.5 A/B/C 공통·일관성 손실
- `06_ocr_diagnostic.zip`: §5.2 학습 전 OCR 사전 검사
- `07_training_and_plan.zip`: §4.7·§5.3–5.4 학습·lambda 탐색·체크포인트
- `08_ocr_and_metrics.zip`: §5.5–5.6 고정 OCR·CER·영상 지표
- `09_results_aggregation.zip`: §5.7–5.9·§6 severity·seed 결과 집계
- `10_case_analysis.zip`: §5.10 대표·심각·C 악화 사례 선정

개별 안내는 `PART_*.md`, 원본 검증값은 `packaging/*.json`, 분리 후 검사는 `validation/*.json`입니다. 예제 설정은 validation pilot 후 확정해야 합니다.

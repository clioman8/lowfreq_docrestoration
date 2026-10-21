# 저주파 consistency 문서 복원: 실험 시작용 Python 프로젝트

첨부 v5의 한 언어·한 U-Net·한 diffusion-reaction 열화군·고정 OCR 설계를 코드로 연결한 v0.1입니다. 이 묶음에는 작동하는 데이터 생성, 수치 연산자, 손실, 학습, OCR 평가, 집계가 들어 있습니다. 본실험 결과나 성능 우위가 들어 있는 완성 논문은 아닙니다.

## 1. 논문의 어디에 어떤 코딩이 필요한가

| 초안 | 구현 파일 | 하는 일 | 연구자가 확정할 것 |
|---|---|---|---|
| §3.4, §5.1 데이터 | `docrestore/data.py` | 책별 텍스트 읽기, 중복·작품 그룹 처리, source 분할, 렌더링, 정확한 GT 저장 | 사용할 영어 책 ID와 작품/판본 그룹, 문자·레이아웃 규칙 |
| §3.2 열화 | `docrestore/operators.py` | 잉크 밀도에 대한 implicit Euler, zero-flux 경계, NumPy/Torch 일치 | h, 시간 스텝 수, train 파라미터 범위 |
| §3.3 강도 변화 | `data.py`, `configs/*.json` | 같은 validation/test 영역에 S0-S4 생성 | 실제 OCR 곡선을 본 뒤 severity 수치 동결 |
| §4.1 모델 | `docrestore/model.py` | 공통 4-level U-Net, 기본 채널 32/64/128/256 | 모든 군에서 같은 아키텍처 유지 |
| §4.2-4.5 손실 | `docrestore/losses.py` | L1+SSIM, full residual, 저주파 투영 residual | beta, m, Fourier 순서, lambda 후보 |
| §4.6-4.7 비교군 | `docrestore/train.py`, `plan.py` | A/B/C, 대응 seed, 동일 lambda 탐색 예산 | pilot 후 학습 예산·선택 규칙 동결 |
| §5.2 사전검사 | `docrestore/diagnose.py` | validation만으로 clean/degraded OCR·시간 확인 | clean 가독성, 유용한 강도 구간 |
| §5.4 체크포인트 | `docrestore/train.py` | validation 최대 shifted CER → 평균 순 선택 | test로 선택하지 않기 |
| §5.5-5.6 OCR | `docrestore/evaluate.py`, `metrics.py` | 고정 Tesseract, CER/PSNR/SSIM, 실행 provenance | 엔진·영어 모델·OEM·PSM·정규화 |
| §5.7-5.9, §6 결과 | `docrestore/aggregate.py` | severity/seed 집계, seed별 최대 CER 후 평균·SD, threshold | 표 해석과 한계 작성 |
| §5.10, §6.5 사례 | `docrestore/cases.py` | input CER 기반 대표·threshold·severe 사례와 C-A 악화 사례의 결정적 선정 | validation에서 인접 구간을 동결하고 그림·해석 작성 |

사례 선택 JSON은 만들 수 있지만 논문용 그림 배치·사례 해석, 문헌 검토, 결과 해석, 최종 원고 수정은 연구자가 수행할 부분입니다. RBF/hat, LM/PPL/NER, 역사자료, 다국어 DB는 포함하지 않습니다.

## 2. 세 원천은 서로 다른 역할이다

| 파일 | 처리 위치 | 학습되는가 |
|---|---|---|
| `librispeech-lm-corpus.tgz` | `data.iter_corpus_books` → 본문 정리 → 책/작품 분할 → 후보 문단 | 텍스트를 영상으로 그린 뒤 복원 모델의 데이터로 사용 |
| `liberation-fonts-ttf-2.1.5.tar.gz` | `assets.prepare_fonts` → Regular 3종과 LICENSE → `render_text_block` | 폰트 자체는 학습하지 않음 |
| `eng.traineddata` | `assets.check_eng` → `evaluate.evaluate_manifest` | OCR 모델은 재학습하지 않음. 평가에만 사용 |

폰트는 `LiberationSans-Regular.ttf`, `LiberationSerif-Regular.ttf`, `LiberationMono-Regular.ttf`입니다. 원문은 정규화된 `librispeech-lm-norm.txt.gz` 대신 원서 ID를 추적할 수 있는 corpus 묶음을 사용합니다. 다운로드 주소와 원천은 `SOURCES.md`에 있습니다.

현재 언어 필터는 Latin 문자 비율 검사일 뿐 영어 판별기가 아닙니다. 본격적인 pilot 전에 영어로 확인한 책의 ID 목록을 `english_books.json`에 배열로 저장하고 `data.book_allowlist`에 지정하십시오. 다른 판본/중복 작품은 `data.work_id_mapping`으로 같은 그룹에 묶을 수 있습니다. `data.language_validation=true`는 이 선정을 실제 확인했다는 사용자 선언이며, 코드가 자동으로 영어임을 증명한다는 뜻이 아닙니다.

## 3. 실행 환경

- Python 3.10 이상.
- `python -m pip install -r requirements.txt`
- Tesseract 엔진은 별도 설치하고 `tesseract --version`이 실행되어야 합니다.
- 고정 영어 모델은 `assets/tessdata_best/eng.traineddata`로 둡니다.
- 기본 연구 설정은 Tesseract 5.5.3 / eng / OEM 1 / PSM 6입니다. 엔진 버전을 바꾸려면 validation 단계에서 선택하고 최종 시험 전에 설정 전체를 고정하십시오.
- `pilot` 실행은 현재 엔진 버전을 기록합니다. `final` 실행은 설정한 엔진 버전, 고정 영어 모델, 동결 상태, 영어 원천 확인을 검사합니다.
- CPU로 기능검사는 할 수 있습니다. 본학습 시간은 실제 GPU/영상 크기에서 pilot으로 측정하십시오. 이 코드의 CPU 검사로 GPU 작동이나 소요시간을 보증하지 않습니다.

## 4. 처음 실행할 순서

아래 명령은 프로젝트 폴더에서, 선택한 Python 환경으로 실행합니다. 원천 파일을 `assets/raw/`에 두면 경로를 그대로 사용할 수 있습니다.

### (1) 원천 파일 확인과 폰트 추출

```bash
python -m docrestore.assets check --corpus assets/raw/librispeech-lm-corpus.tgz --eng assets/raw/eng.traineddata
python -m docrestore.assets fonts --archive assets/raw/liberation-fonts-ttf-2.1.5.tar.gz --output assets/fonts
```

`assets/tessdata_best/` 폴더를 만들고 검증한 `eng.traineddata`를 그 안에 복사합니다. Windows PowerShell의 `START_WINDOWS.ps1`은 환경·원천 검증·폰트 추출·pilot 데이터 생성을 묶은 시작 스크립트입니다. 영어 책 allowlist를 설정한 뒤 사용하는 것이 좋습니다. 스크립트는 Tesseract를 설치하거나 장시간 학습을 시작하지 않습니다.

### (2) 작은 D0 생성

`configs/pilot.example.json`의 `data.book_allowlist`와 `language_validation`을 먼저 정합니다. 예제 강도·개수·학습 step은 검증되지 않은 값입니다.

```bash
python -m docrestore.data --corpus assets/raw/librispeech-lm-corpus.tgz --fonts-dir assets/fonts --output data/pilot --config configs/pilot.example.json --prepare-eval
```

생성물:

- `manifest.jsonl`: clean 영역과 실제로 그린 정답 텍스트.
- `split_manifest.json`: 책/작품별 분할과 원천 해시.
- `dataset_metadata.json`: 생성 설정·선정 원서 수·영상 수.
- `eval_manifest.jsonl`: validation/test의 S0-S4 대응.
- `clean/`: clean PNG.
- `degraded_arrays/` 또는 manifest에 기록된 `.npy`: 복원 네트워크가 받는 float32 관측.
- `degraded/`: 같은 관측의 8-bit PNG. OCR 입력/참조용.

기존 데이터는 기본적으로 덮어쓰지 않습니다. 생성에 실패하면 원문 풀과 오류를 확인한 뒤 별도의 출력 폴더에서 다시 시작하십시오.

### (3) 학습 전에 validation-only OCR 진단

```bash
python -m docrestore.diagnose --data-root data/pilot --config configs/pilot.example.json --tessdata-dir assets/tessdata_best --output-dir runs/pilot_diagnostic
```

이 명령은 test OCR을 실행하지 않습니다. clean부터 인식이 나쁜지, S0-S4가 난도 차이를 만드는지, OCR 시간이 어느 정도인지 확인합니다. 텍스트/폰트/강도 조정은 이 단계에서 끝내십시오.

### (4) A/B/C 소규모 학습

```bash
python -m docrestore.train fit --data-root data/pilot --config configs/pilot.example.json --group A --seed 0 --run-dir runs/pilot_A_seed0 --tessdata-dir assets/tessdata_best
python -m docrestore.train fit --data-root data/pilot --config configs/pilot.example.json --group B --lambda-value 0.1 --seed 0 --run-dir runs/pilot_B_seed0 --tessdata-dir assets/tessdata_best
python -m docrestore.train fit --data-root data/pilot --config configs/pilot.example.json --group C --lambda-value 0.1 --seed 0 --run-dir runs/pilot_C_seed0 --tessdata-dir assets/tessdata_best
```

각 run은 `best.pt`, `last.pt`, `summary.json`, `run_metadata.json`, `training_log.jsonl`, best validation 이미지·OCR를 남깁니다. 한 run 폴더를 다른 실험으로 덮어쓰지 않습니다. 현재 starter는 재개 학습(resume)을 구현하지 않았으므로 장시간 예산은 pilot에서 먼저 잡으십시오.

`training_log.jsonl`에서 weighted consistency / data gradient 비율과 투영 잔차 에너지 비율을 확인합니다. 같은 숫자 lambda가 같은 효과를 뜻하지 않습니다. 현재 선형 matched 연산에서는 full consistency도 이미 주파수 가중을 포함합니다. 학습이 끝나기 전에 C의 우위를 가정하지 마십시오.

### (5) source split 유지, 설정 동결, lambda와 seed 반복

최종 데이터 규모를 늘릴 때 영어 원서 목록·작품 그룹·책별 split을 바꾸면 pilot에서 본 문서가 최종 test에 들어갈 수 있습니다. 처음부터 최종 원천 풀을 정하고, `data.source_split_manifest`에 최초 `data/pilot/split_manifest.json`을 지정하여 분할을 재사용합니다. pilot/final의 원천 ID 집합과 텍스트 해시가 달라지면 이 경로는 중단하도록 설계했습니다.

`configs/final.template.json`은 출발점입니다. 숫자를 실제 측정에 맞춰 고친 뒤 `configuration_frozen=true`, `calibration_status=FROZEN`으로 두십시오. 이 표시는 실제 검토를 대신하지 않습니다. 최종 데이터는 새 폴더로 생성합니다.

```bash
python -m docrestore.plan --config configs/final.template.json --data-root data/final --tessdata-dir assets/tessdata_best --runs-root runs/final --stage final --output runs/final_plan.json
```

계획 JSON에는 shell 문자열 대신 argv 배열이 있습니다. 자동으로 학습을 실행하지 않습니다. 기본안은 seed 0에서 A 1개, B/C 각 lambda 3개로 총 7회 탐색한 후, B/C의 선택된 seed-0 run을 재사용하고 나머지 seed 1/2를 실행하는 방식입니다. 이 구체적인 규칙에서는 **실제 13회 실행, 최종 표에는 9회 반복**이 들어갑니다. 모든 후보는 같은 학습·validation 예산을 사용합니다. 탐색과 최종 반복을 분리하려면 실행 수가 달라집니다.

phase1이 끝나면 `plan`에 `--select-from runs/final`을 추가하여 validation 기록만으로 나머지 실행 목록을 만들 수 있습니다. 후보의 설정/그룹/seed/lambda가 맞는지도 검사합니다.

### (6) 동결된 best 모델로 시험 평가

```bash
python -m docrestore.train predict --checkpoint runs/final/C_lambda0p1_seed0/best.pt --manifest data/final/eval_manifest.jsonl --split test --data-root data/final --output-dir runs/final_test/C_seed0
python -m docrestore.evaluate --manifest runs/final_test/C_seed0/predictions_manifest.jsonl --images-root data/final --output runs/final_test/C_seed0_ocr.jsonl --mode restored --tessdata-dir assets/tessdata_best --method C --seed 0
```

위 checkpoint 경로의 lambda는 예시입니다. 실제 선택된 경로를 사용하고 A/B/C와 모든 seed에 반복합니다. clean/degraded 참조 역시 **같은 test sample 목록만** 별도 manifest로 골라 평가합니다. `evaluate`에 전체 eval manifest를 넣으면 validation과 test가 함께 평가되므로, pilot에서는 반드시 `diagnose`를 사용하십시오.

```bash
python -m docrestore.aggregate --results runs/final_test/C_seed0_ocr.jsonl --config configs/final.template.json --output-dir runs/final_tables --plot
```

`--results` 뒤에 A/B/C 전체 seed와 참조 OCR JSONL 경로를 나열하십시오. 단일 파일 명령은 사용법 예시이며, 3-seed 비교를 대체하지 않습니다. 최대값은 **severity별 corpus CER → seed별 최대 → seed 평균·표본 SD** 순으로 집계합니다. 문서/GT 불일치, 혼합 OCR 설정, 같은 run의 서로 다른 checkpoint 혼합을 검사합니다.

```bash
python -m docrestore.cases --A runs/final_test/A_seed0_ocr.jsonl --B runs/final_test/B_seed0_ocr.jsonl --C runs/final_test/C_seed0_ocr.jsonl --degraded runs/final_test/degraded_ocr.jsonl --config configs/final.template.json --output runs/final_tables/cases.json
```

`analysis.example_seed`와 `threshold_adjacent_interval`을 사전에 정합니다. 근접 사례나 C가 A보다 악화한 사례가 없으면 없음으로 기록합니다. 억지로 실패·복원 성공 사례를 만들어내지 않습니다.

## 5. 수학·측정 명세

- 영상은 검정 글자/흰 배경, ink density `v=1-u`.
- 대칭 zero-flux graph Laplacian을 DCT-II로 대각화합니다. DCT로 계산하는 것은 각 implicit Euler 단계의 같은 선형계 해이며 별개의 열화 모델이 아닙니다.
- 실제 공간 간격 기본값은 pixel 단위 `h=1`. 해상도 변경 후 물리적 의미를 그대로 가정하지 마십시오.
- 네트워크 입력은 `y`만 받습니다. eta는 학습쌍 생성과 forward consistency 계산에 사용합니다.
- Fourier 기저는 Neumann solver의 고유기저와 다를 수 있습니다. solver에 주기 경계를 강요하지 않습니다.
- 기저는 DC 포함, 반경 정렬, cos→sin 순서, 정확히 m개. m=32에서는 마지막 cos/sin 쌍이 잘릴 수 있는 명시적 관례입니다. 선택한 관례를 원고에 적으십시오.
- full/subspace 모두 pixel 수 N으로 나눕니다. subspace 에너지를 m으로 나누지 않습니다.
- SSIM은 Gaussian 11×11, sigma 1.5, population moments, VALID 영역 평균, range=1. 작은 QA 영상은 들어가는 홀수 창을 사용합니다. 모델은 nearest upsampling을 사용합니다.
- 학습/복원용 관측은 float, OCR과 보고용 이미지 지표는 저장된 uint8 PNG 기준입니다. 모든 방법에 같은 round/clip 규칙을 적용합니다.
- CER는 NFC+공백 정규화, 대소문자·구두점 유지. 전체 edit distance / 전체 GT 길이이며 1로 자르지 않습니다.
- threshold crossing은 보조 지표이며, 처음부터 초과/범위 내 미도달을 임의 숫자로 평균하지 않습니다.
- noise-free matched 조건의 consistency는 새 정답 정보를 추가하는 것이 아니라 복원오차 가중을 바꿉니다. 유리한 결과뿐 아니라 무효·악화 결과도 보고해야 합니다.

## 6. 검증과 현재 한계

```bash
python -m unittest discover -s tests -v
python examples/smoke.py --fonts-dir assets/fonts --tessdata-dir assets/tessdata_best --output runs/smoke
```

smoke는 직접 작성한 영어 fixture로만 실행합니다. SLR11 성능 실험이 아니며 validation만 사용합니다. A/B/C 3-step 학습, OCR, 집계, best checkpoint 재로딩을 연결하는 기능검사입니다. `examples/validation_report.json`에 이 환경에서 실제 검사한 범위를 기록했습니다.

실제 원문 archive 전체 검사·영어 원서 선정·main GPU 학습·3 seed 성능 검증·실험에 맞춘 논문 결과 작성은 아직 수행하지 않았습니다. 예제 severity, lambda, 학습 step을 검증된 설정으로 인용하지 마십시오.

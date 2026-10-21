# 03_severity_generation: 동일 문서 S0–S4 생성

초안 대응: **§3.3**

주 구현: `docrestore/data.py`.

보조 모듈: `assets.py`, `operators.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/03_severity_generation.txt
```

## 입력

01번과 같은 원천. configs/pilot.example.json의 degradation.severities와 n_steps, h.

## 실행

```bash
python -m docrestore.assets fonts --archive assets/raw/liberation-fonts-ttf-2.1.5.tar.gz --output assets/fonts
python -m docrestore.data --corpus assets/raw/librispeech-lm-corpus.tgz --fonts-dir assets/fonts --output data/pilot --config configs/pilot.example.json --prepare-eval
```

## 출력과 구현 범위

clean 데이터와 validation/test의 eval_manifest.jsonl, S0–S4 PNG, float32 관측 NPY, 각 표본의 열화 파라미터.

01번과 같은 data.py의 --prepare-eval 경로입니다. 같은 문서 ID에 모든 강도를 대응시키며 S0도 별도 열화 조건입니다. clean truth는 clean 영상입니다. 예제 강도는 validation OCR로 보정한 뒤 동결해야 합니다. 출력 폴더는 새 경로를 사용합니다.

## 검증

python -m unittest discover -s tests -v

분리된 파일의 검사 결과: `validation/03_severity_generation.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

# 01_data_preparation: 데이터 원천·분할·렌더링

초안 대응: **§3.4·§5.1**

주 구현: `docrestore/data.py`.

보조 모듈: `assets.py`, `operators.py`.

## 설치

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```bash
python -m pip install -r requirements/01_data_preparation.txt
```

## 입력

SLR11 원문 archive, Liberation Fonts archive. 영어로 확인한 원서 ID를 config의 data.book_allowlist로 지정합니다.

## 실행

```bash
python -m docrestore.assets fonts --archive assets/raw/liberation-fonts-ttf-2.1.5.tar.gz --output assets/fonts
python -m docrestore.data --corpus assets/raw/librispeech-lm-corpus.tgz --fonts-dir assets/fonts --output data/pilot --config configs/pilot.example.json
```

## 출력과 구현 범위

clean PNG, 정확히 렌더링한 GT, manifest.jsonl, split_manifest.json, dataset_metadata.json.

책·작품 단위 분할과 정규화된 동일 문단 제거를 수행합니다. 의미가 유사한 문장이나 번역본까지 자동으로 제거하는 기능은 아닙니다. 영어 책 선정을 자동으로 확정하지 않습니다.

## 검증

python -m unittest discover -s tests -v

분리된 파일의 검사 결과: `validation/01_data_preparation.json`. 전체 흐름: `docs/FULL_EXPERIMENT_GUIDE.md`.

# Primary sources and exact asset choices

This project implements the supplied v5 protocol. No external repository code was copied.

- Supplied draft SHA-256: `483eeaffdfe9d542e72fc0efe019c063b41717ef4d8ad0fa4b9285b2bbf734b3`.
- SLR11 official description and license: https://www.openslr.org/11/
- SLR11 archive: https://openslr.elda.org/resources/11/librispeech-lm-corpus.tgz
- Official checksums: https://www.openslr.org/resources/11/checksum.md5
- Book-file layout corroboration: https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/lm/normalize_text.sh
- Liberation Fonts binary 2.1.5 release: https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5
- Binary archive: https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz
- Font license: https://github.com/liberationfonts/liberation-fonts/blob/2.1.5/LICENSE
- Font filenames: https://github.com/liberationfonts/liberation-fonts/blob/2.1.5/Makefile
- Fixed OCR model: https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/e12c65a915945e4c28e237a9b52bc4a8f39a0cec/eng.traineddata
- Model repository/license: https://github.com/tesseract-ocr/tessdata_best
- Tesseract OEM/PSM documentation: https://tesseract-ocr.github.io/tessdoc/Data-Files.html and https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- Planned engine release: https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3

The code archive does not redistribute the source corpus, font archive, or OCR model. Font extraction preserves the accompanying OFL license. Retain original dataset/model licenses when redistributing research assets.

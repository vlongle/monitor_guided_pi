gemini-2.5-flash ~3s, good
QWen3-VL-8B-Instruct ~21s, good
QWen3-VL-2B-Instruct, ~20s, good
Siglip,  49.86it/s



QWen2.5-VL-7B-Instruct is trash.


Down-resize image -> QWen now takes 14s.


```
pip install -e .
cd vlmx
pip install -e .
MAX_JOBS=16 pip install -v -U flash-attn --no-build-isolation
```
export PYTHONPATH=.

.PHONY: run-faster-rcnn run-yolo infer-yolo app

run-faster-rcnn:
	uv run python src/faster_rcnn/model.py

run-yolo:
	uv run python src/yolo11/yolo11_train.py --device mps
eval-yolo:
	uv run python src/yolo11/eval.py
infer-yolo:
	uv run python src/yolo11/inference.py $(ARGS)

run-inference-rcnn:
	uv run python src/faster_rcnn/inference.py

app:
	uv run streamlit run python app.py
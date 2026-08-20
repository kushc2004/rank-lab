.PHONY: install audit features baselines full report dashboard test kaggle-push kaggle-watch

install:
	python -m pip install -e '.[full,dev]'

audit:
	python scripts/audit_kuairand.py data=kuairand_pure

features:
	python scripts/build_features.py data=kuairand_pure

baselines:
	python scripts/run_cached_baselines.py data=kuairand_pure

full:
	python scripts/run_full_pipeline.py device=auto

report:
	python scripts/export_report.py run_id=latest

dashboard:
	streamlit run app/streamlit_app.py

test:
	pytest

kaggle-push:
	kaggle kernels push -p notebooks

kaggle-watch:
	python3 scripts/watch_kaggle_kernel.py --slug kushchaudhari/ranklab-kuairand-pure-full-pipeline

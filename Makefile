.PHONY: data features train backtest serve dashboard all

data:
	python src/data/fetch_macro.py
	python src/data/social.py

features:
	python src/features/engineering.py

train:
	python src/train/train.py

backtest:
	python src/backtest/backtest.py

serve:
	uvicorn src.serve.app:app --reload

dashboard:
	streamlit run src/serve/dashboard.py

all: data features train backtest

.PHONY: jupyter webapp

jupyter:
	hatch run jupyter:jupyter lab

webapp:
	streamlit run src/birds/web_app.py

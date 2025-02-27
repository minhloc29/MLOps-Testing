install:
	pip install --upgrade pip %%\
	pip install -r requirements.txt

test:
	python -m pyteset -vvv --cov=hello --cov=greeting \
		--cov=smath --cov=web tests

debug:
	python -m pytest --v --pdb

one-test:

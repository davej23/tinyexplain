isort:
	isort .
mypy:
	python -m mypy --config-file=mypy.ini tinyexplain/
pylint:
	python -m pylint --rcfile pylintrc tinyexplain/
black:
	python -m black --config pyproject.toml .
run_tests:
	python -m pytest --cov-config=.coveragerc --cov=tinyexplain

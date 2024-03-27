isort:
	isort .
mypy:
	python -m mypy --config-file=mypy.ini tinyexplain/
pylint:
	python -m pylint tinyexplain/
black:
	python -m black tinyexplain/
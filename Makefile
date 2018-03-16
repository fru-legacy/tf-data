upload:
	python3 setup.py sdist &&\
	twine upload dist/*

tag:
	git tag 0.0.1 -m "0.0.1 - Empty release" &&\
	git push --tags origin master

register:
	python setup.py register -r pypi
upload:
	python setup.py sdist upload -r pypi
tag:
	git tag 0.0.1 -m "0.0.1 - Empty release" &&\
	git push --tags origin master

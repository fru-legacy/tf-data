upload:
	python3 setup.py sdist
	twine upload dist/*

tag:
	git tag 0.0.2 -m "0.0.2 - Second empty release"
	git push --tags origin master

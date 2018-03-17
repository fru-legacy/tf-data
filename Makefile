publish:
	python3 setup.py sdist
	twine upload dist/*

tag:
	git tag 0.0.4 -m "0.0.4"
	git push --tags origin master

cleanup:
	rm -rf dist
	rm -rf tf_data.egg-info

upload-dataset:
	drive push ~/datasets/tf-data
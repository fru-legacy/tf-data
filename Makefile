publish:
	python3 setup.py sdist
	twine upload dist/*

tag:
	git tag 0.0.3 -m "0.0.3"
	git push --tags origin master
	make cleanup

cleanup:
	rm -rf dist
	rm -rf tf_data.egg-info

upload-dataset:
	drive push ~/datasets/tf-data
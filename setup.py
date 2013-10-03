	#!/usr/bin/env python

	from distutils.core import setup, Extension
	import numpy.distutils.misc_util


	py_modules = []

		setup(
			name="faststats",
			version='0.0dev',
			author="Morgan Fouesneau",
			author_email="mfouesn@uw.edu",
			py_modules=py_modules,
			description="",
			long_description=open("README.md").read(),
			classifiers=[
				"Development Status :: 0 - Beta",
				"Intended Audience :: Science/Research",
				"Operating System :: OS Independent",
				"Programming Language :: Python",
			    ],
		    )

		# python setup.py build_ext --inplace

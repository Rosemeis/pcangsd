from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"pcangsd.reader_cy",
		["pcangsd/reader_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
		language="c++"
	),
	Extension(
		"pcangsd.shared_cy",
		["pcangsd/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"pcangsd.covariance_cy",
		["pcangsd/covariance_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"pcangsd.inbreed_cy",
		["pcangsd/inbreed_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"pcangsd.admixture_cy",
		["pcangsd/admixture_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"pcangsd.tree_cy",
		["pcangsd/tree_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="pcangsd",
	version="1.35",
	author="Jonas Meisner",
	description="Framework for analyzing low depth NGS data in heterogeneous populations using PCA",
	packages=["pcangsd"],
	entry_points={
		"console_scripts": ["pcangsd=pcangsd.pcangsd:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		'numpy',
		'cython',
		'scipy'
    ],
    ext_modules=cythonize(extensions, compiler_directives={'language_level':'3'}),
    include_dirs=[numpy.get_include()]
)

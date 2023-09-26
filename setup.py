from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"src.reader_cy",
		["src/reader_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
		language="c++"
	),
	Extension(
		"src.shared_cy",
		["src/shared_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"src.covariance_cy",
		["src/covariance_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"src.inbreed_cy",
		["src/inbreed_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"src.admixture_cy",
		["src/admixture_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"src.tree_cy",
		["src/tree_cy.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="pcangsd",
	version="1.2",
	author="Jonas Meisner",
	description="Framework for analyzing low depth NGS data in heterogeneous populations using PCA",
	packages=["src"],
	entry_points={
		"console_scripts": ["pcangsd=src.pcangsd:main"]
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

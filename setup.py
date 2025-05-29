from setuptools import setup, find_packages

setup(
    name='geneML',
    version='0.1.0',
    description='Gene Annotation Across Diverse Fungal Species Using Deep Learning',
    author='Lawrence Hon',
    author_email='lhon@hexagonbio.com',
    url='https://github.com/hexagonbio/geneml',
    packages=find_packages(),
    install_requires=[
        'biopython>=1.78',
        'tensorflow>=2.17.0',
        'numpy>=1.21.0',
        'numba>=0.53.1',
        'tqdm>=4.62.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'geneml=geneml.main:main',
        ],
    },
)

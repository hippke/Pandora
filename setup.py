from setuptools import setup
from os import path

exec(open(path.join("pandoramoon", 'version.py')).read())

# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install wotan). Irrelevant for users using Python2
try:
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

setup(
    name='pandoramoon',
    version=PANDORA_VERSIONING,
    description='Exomoon transit detection algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hippke/pandora',
    author='Michael Hippke',
    author_email='michael@hippke.org',
    license='GPL3',
    zip_safe=False,
    include_package_data=True,
    package_data={'': ['*.csv', '*.cfg']},
    packages=['pandoramoon'],
    install_requires=[
        'numpy',
        'numba',
        'tqdm',
        'matplotlib'
        ]
)
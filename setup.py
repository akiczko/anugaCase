from setuptools import setup, find_packages

setup(
    name='anugaTools',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'anuga>=3.2.0', 'geopandas', 'matplotlib', 'numpy', 'shapely', 'gdal'  # From your imports
    ],
    description='Tools for parsing and creating ANUGA 2D models',
    author='Your Name',
    author_email='akiczko@gmail.com',
    url='https://github.com/akiczko/anugaCase',
)
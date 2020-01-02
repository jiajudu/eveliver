from setuptools import setup


if __name__ == '__main__':
    # To upload this package to pypi:
    # python setup.py sdist
    # python -m twine upload dist/*
    setup(
        name='eveliver',
        version='1.1.0',
        description='Some pytorch utilities for NLP',
        author='Jiaju Du',
        author_email='i@dujiaju.me',
        url='https://github.com/jiajudu/evelivers',
        packages=['eveliver'],
        install_requires=[
            'torch>=1.2.0',
            'numpy>=1.16.4'
        ]
    )

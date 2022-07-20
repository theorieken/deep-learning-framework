from setuptools import setup, find_packages


setup(
    name='deep_learning_framework',
    version='0.1.5',
    license='MIT',
    author="Theo Rieken",
    author_email='mail@theorieken.de',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/theorieken/deep-learning-framework',
    keywords='deep learning',
    install_requires=[
        'torch',
        'cycler',
        'matplotlib',
        'numpy',
        'pandas',
        'wandb',
        'torchvision',
        'scipy',
        'plotly',
        'scikit-learn',
    ],
)

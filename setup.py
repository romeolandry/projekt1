from setuptools import setup

setup(name='windml',
      version='0.1',
      packages=['windml'],
      scripts=['windml/multi.py'],
      include_package_data=True,
      install_requires=[
          'joblib',
          'scikit-learn',
      ],
)

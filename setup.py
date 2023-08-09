from setuptools import setup

setup(
    name="HousingPricePredictions",
    version="0.1",
    author="Sahil khanna",
    author_email="sahil.khanna@tigeranalytics.com",
    license="LICENSE.txt",
    packages=["HousingPricePredictions"],
    package_dir={"HousingPricePredictions": "src/HousingPricePredictions"},
    description="",
    long_description=open("README.md").read(),
)

from setuptools import setup, find_packages

setup(
    name="mle_training_bhujith",
    version="0.0.1",
    description="Helps you to predict housing prices using Random Forest model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bhujith Madav Velmurugan",
    author_email="bhujith.velmurug@tigeranalytics.com",
    url="https://github.com/bhujithtiger/mle-training",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "pytest", "scikit-learn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)

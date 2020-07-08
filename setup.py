import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="marvin",
    version="0.0.1",
    author="Quinlan Sykora",
    author_email="quinsykora@gmail.com",
    description="The code for the multi-agent routing value iteration network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uber/MARVIN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["tqdm==4.36.1",
                      "torch==1.4.0",
                      "tensorboardX==1.9",
                      "numpy==1.18.1",
                      "cvxpy==1.1.0a1",
                      "scipy==1.4.1"],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["marvin/utils/optimized.so",
             "marvin/utils/LKH",
             "marvin/data/*.pkl"],
    },
)
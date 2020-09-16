import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="knutils",
    version="1.0.0",
    author="Knut Andreas Kvåle",
    author_email="knut.a.kvale@gmail.com",
    description="Knut's Utilities for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knutankv/knutils",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
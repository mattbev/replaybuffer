"""
PyPI build file
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

docs_extras=[
    'sphinx',
    'docutils'
]

setuptools.setup(
    name="<insert distribution name>",
    version="0.0.1",
    author="<insert author>",
    author_email="matt@nodarsensor.com",
    description="<insert short description>",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<insert url>",
    keywords=[],
    install_requires=[
    ],
    extras_require={
        'docs': docs_extras
    },
    # license="MIT",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
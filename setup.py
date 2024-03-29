import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uk",
    version="0.0.1",
    author="Vol K",
    author_email="vol@wilab.org.ua",
    description="uk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/proger/uk",
    packages=setuptools.find_namespace_packages(include=['uk', 'uk.*']),
    python_requires=">=3.7",
    install_requires=[
        "datasets",
        "kaldiio",
        "loguru",
        "praatio<6.0.0",
        "soundfile",
        "sqlite_utils",
        "torch",
        "torchaudio",
        "tqdm",
        "ukro-g2p"
    ]
)

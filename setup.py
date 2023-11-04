import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dadmatools",
    version="1.5.2",
    author="Dadmatech AI Company",
    author_email="info@dadmatech.ir",
    description="DadmaTools is a Persian NLP toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dadmatech/DadmaTools",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "bpemb>=0.3.3",
        "nltk",
        "folium>=0.2.1",
        "spacy>=3.0.0",
        "torch>=1.7.1",
        "transformers>=4.9.1",
        "h5py>=3.3.0",
        "Deprecated==1.2.6",
        "hyperopt>=0.2.5",
        "pyconll>=3.1.0",
        "pytorch-transformers>=1.1.0",
        "segtok>=1.5.7",
        "tabulate>=0.8.6",
        "supar==1.1.2",
        "gensim>=3.6.0",
        "conllu",
        "gdown>=4.3.1",
        # "NERDA",
        "py7zr>=0.17.2",
        "html2text",
        "tf-estimator-nightly==2.8.0.dev2021122109",
        "scikit-learn>=0.24.2",
        "Keras==2.4.3",
        "Keras-Preprocessing==1.1.0",
        "numpy==1.16.4",
        "pandas==0.24.2",
        "pandas-datareader==0.8.1",
        "stanza==1.1.1",
        "transformers==4.1.1",
        "tensorflow==2.2.0",
        "tokenizers==0.9.4",
        "hazm==0.7.0"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License ",
        "Operating System :: OS Independent",
    ],
)

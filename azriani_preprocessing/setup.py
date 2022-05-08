import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "azriani-Ufea-wrangling-4",
    version = "0.0.1",
    author = "Rani",
    autor_email = "azriani.m@gmail.com",
    description = "data wrangling - tugas 4",
    long_description = "preprocessing for autos.csv",
    long_description_content_type = "text/markdown",
    url = "",
    packages = setuptools.find_packages(),
    classifier = ["Programming Language :: Python :: 3"],
    install_requires = [
        "matplotlib",
        "pandas == 1.1.4",
        "scikit-learn == 0.22", #(>= 3.5)
        "seaborn == 0.11.0"],
    python_requires = ">=3.7"
)
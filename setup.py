from setuptools import setup, Extension, find_packages
import os

guidedlda_ext = Extension(
    name="STED.guidedlda._guidedlda",  # 编译后的完整包路径
    sources=[
        "STED/guidedlda/_guidedlda.c",
        "STED/guidedlda/gamma.c"
    ],
    include_dirs=["STED/guidedlda"],  # 头文件(.h)路径
)

setup(
    name="STED",
    version="0.1.1",
    author="Liao Yunxi",
    author_email="2211110091@pku.edu.cn",
    description="Single-cell Topic Modeling & Epigenetic Deconvolution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiaoYunxi/STED",
    packages=find_packages(),  
    include_package_data=True,
    ext_modules=[guidedlda_ext],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "pandas",
        "numpy",
        "networkx",
        "anndata",
        "plotly",
        "scipy",
        "scikit-learn",
        "h5py",
        "matplotlib",
        "tables",
        "tqdm",
        "scanpy",
        "umap-learn"
    ],
)
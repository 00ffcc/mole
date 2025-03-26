from setuptools import setup, find_packages

setup(
    name="embedding_offload",
    version="0.1.0",  
    description="embedding_offload",
    author="00ffcc",  
    author_email="guizhiyu@mail.ustc.edu.cn",
    packages=["embedding_offload"],
    package_dir={"embedding_offload": "embedding_offload"},
    install_requires=[
    ],
    entry_points={
    },
    include_package_data=True, 
)
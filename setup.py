from setuptools import setup, find_packages
from pathlib import Path

VERSION = '0.0.1'
DESCRIPTION = 'A package for jailbreak defense'
requirements = Path('requirements.txt').read_text().splitlines()

setup(
    name="jailbreak-defense",
    version=VERSION,
    author="Qirui Liu",
    author_email="931826123@qq.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=requirements,
    keywords=["Large Language Model", "LLM", "jailbreak defense"],
)
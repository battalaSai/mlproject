from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace("/n","") for req in requirements]
        hypen="-e ."
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements


setup(
name='mlproject',
version='0.0.1',
author='Sai Rohith',
packages=find_packages(),

requires=get_requirements('requirements.txt')
)
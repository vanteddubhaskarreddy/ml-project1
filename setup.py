from setuptools import find_packages,setup
# here we are importing the find_packages and setup from the setuptools module
# find_packages is used to find the packages in the current directory
# setup is used to setup the package and install it
from typing import List

def get_requirements(file_path: str) -> List[str]:
    req = []
    with open(file_path) as file_object:
        req = file_object.read().splitlines()
        req = [ele.replace('\n', '') for ele in req]

        if '-e .' in req:
            req.remove('-e .')
    
    return req

# we are using the setup function to setup the package and install it
setup(
name = 'ml-project1',# name of the package
version = '0.0.1', # version of the package
author = 'Bhaskar Reddy', # author of the package
author_email = 'vanteddubhaskarreddy@gmail.com', # author email
packages = find_packages(), # finding the packages in the current directory
install_requires = get_requirements('requirements.txt') # installing the required packages

)

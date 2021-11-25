from setuptools import setup, find_packages

with open("requirements.txt") as requirements_file:
    install_requires = [line.strip() for line in requirements_file]

setup(name='code_transformer',
      version='0.1',
      description='Code Transformer',
      author='Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de,kirschto@in.tum.de',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False)

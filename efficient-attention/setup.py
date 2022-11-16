from setuptools import setup

install_requires = [x.strip('\n') for x in open("requirements.txt", 'r').readlines()]

setup(
    name='efficient_attention',
    version='0.1',
    description='efficient attention package',
    author='Jiangtao Feng',
    install_requires=install_requires,
    packages=['efficient_attention'],
)
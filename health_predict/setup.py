# -*- coding: utf-8 -*-
import io
from setuptools import setup

#with io.open('ReadMe.md','rt',encoding='utf8') as f:
#    readme = f.read()

setup(
    name='healthpredict', #pypi中的名称，pip或者easy_install安装时使用的名称
    version='0.1.2',
    #url='https://www.palletsprojects.com/p/flask/',
    #project_urls=OrderedDict((
        #('Documentation', 'http://flask.pocoo.org/docs/'),
        #('Code', 'https://github.com/pallets/flask'),)),
    #license='BSD',
    author='sxx',
    author_email='sxx@dfc.com',
    maintainer='sxx',
    maintainer_email='sxx@dfc.com',
    description='A simple tool for predicting health score.',
   # long_description=readme,
    packages=['health_predict'],#代码import package名称
    package_data = {
            'health_predict':['data/*.csv','data/*.hdf5','models/*.h5']},
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=[ # 需要安装的依赖
        'numpy>=1.9.1',
        'pandas>=0.22.0',
        'tensorflow>=1.6.0',
        'keras>=2.1.5',
    ],
    classifiers=[ # 程序的所属分类列表
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    entry_points={
        'console_scripts': [
            'test = health.test:main',
        ],
    },
)
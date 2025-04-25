from setuptools import setup, find_packages

setup(
    name='deepmhc',
    version='1.0.1',
    description='DeepMHC: Peptide MHC binding and presentation prediction for class I and II',
    packages=find_packages(exclude=['results']),
    exclude_package_data={"deepmhc": ["main.py"]},
    include_package_data=True,
    author='Jinxing Li, Wei Qu, Ronghui You, Shanfeng zhu',
    author_email='jinxingli23@m.fudan.edu.cn, zhusf@fudan.edu.cn',
    url='',
    download_url='',
    keywords=['Immunology', 'Peptidomics'],
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    scripts=[
            #  'bin/deepmhci',
             'bin/deepmhcii',
             ],
    test_suite="nose.collector",
    tests_require=['nose'],
    install_requires=[
                        'scipy==1.10.1',
                        'scikit-learn==1.0.2',
                        'click==8.0.4',
                        'ruamel.yaml==0.16.12',
                        'tqdm==4.62.3',
                        'logzero==1.7.0',
                        'torch==1.13.1'
                      ],
)

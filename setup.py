import setuptools


setuptools.setup(
    name='tgnews',
    version='0.0.1',
    author='Ivan Krivosheev',
    author_email='py.krivosheev@gmail.com',
    description='Telegram news',
    zip_safe=False,
    python_requires='>=3.5',
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'tgnews=tgnews.main:__main__',
        ]
    }
)

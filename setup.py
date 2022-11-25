from setuptools import setup
from glob import glob

package_name = 'EE_133a_Project'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz', glob('rviz/*')), 
        ('share/' + package_name + '/urdf', glob('urdf/*')), 
        ('share/' + package_name + '/launch', glob('launch/*')), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='EE 133a final project',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'beat_saber = EE_133a_Project.beat_saber:main'
        ],
    },
)

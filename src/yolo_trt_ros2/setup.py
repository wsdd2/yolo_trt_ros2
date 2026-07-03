from setuptools import setup

package_name = 'yolo_trt_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.backends'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/yolo_detector.launch.py',
            'launch/inspection_perception.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/detector.yaml',
            'config/inspection_perception.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MscapeTech',
    maintainer_email='TODO@example.com',
    description='Lightweight ROS2 Foxy YOLO detector node for Jetson Orin NX.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector_node = yolo_trt_ros2.yolo_detector_node:main',
            'coordinate_projector_node = yolo_trt_ros2.coordinate_projector_node:main',
        ],
    },
)

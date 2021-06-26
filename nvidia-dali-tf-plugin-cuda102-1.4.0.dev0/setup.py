# Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_namespace_packages
from dali_tf_plugin_install_tool import InstallerHelper
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
from setuptools.extension import Extension
import os

class CustomBuildExt(build_ext, object):
    def run(self):
        helper = InstallerHelper(plugin_dest_dir = os.path.join(self.build_lib, 'nvidia', 'dali_tf_plugin'))
        helper.install()

class CustomDistribution(Distribution):
    def __init__(self, attrs=None):
        Distribution.__init__(self, attrs)
        # Just telling distutils that we have an ext module
        # It doesn't matter what we write here, because we are overriding the
        # build_ext step altogether.
        # By filling this ext_modules we are signaling that this package needs
        # to be built for different platforms
        self.ext_modules = [Extension('nvidia.dali_tf_plugin', [])]

setup(name='nvidia-dali-tf-plugin-cuda102',
      description='NVIDIA DALI  Tensorflow plugin for CUDA 10.2. Git SHA: 5cb3fe114cfc4986098a765374ef6eee117c15a3',
      url='https://github.com/NVIDIA/dali',
      version='1.4.0dev',
      author='NVIDIA Corporation',
      license='Apache License 2.0',
      packages=find_namespace_packages(include=['nvidia.*']),
      include_package_data=True,
      zip_safe=False,
      install_requires = [
          ],

      cmdclass={
          'build_ext': CustomBuildExt,
      },
      distclass=CustomDistribution
     )

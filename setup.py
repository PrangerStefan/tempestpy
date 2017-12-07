import os
import sys
import subprocess
import datetime

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test
from distutils.version import StrictVersion

import setup.helper as setup_helper
from setup.config import SetupConfig

if sys.version_info[0] == 2:
    sys.exit('Sorry, Python 2.x is not supported')

# Minimal storm version required
storm_min_version = "1.1.0"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='', subdir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.subdir = subdir


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ('storm-dir=', None, 'Path to storm root (binary) location'),
        ('disable-dft', None, 'Disable support for DFTs'),
        ('disable-pars', None, 'Disable support for parametric models'),
        ('debug', None, 'Build in Debug mode'),
        ('jobs=', 'j', 'Number of jobs to use for compiling'),
    ]

    config = SetupConfig()

    def _extdir(self, extname):
        return os.path.abspath(os.path.dirname(self.get_ext_fullpath(extname)))

    def run(self):
        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # Build cmake version info
        build_temp_version = self.build_temp + "-version"
        setup_helper.ensure_dir_exists(build_temp_version)

        # Write config
        setup_helper.ensure_dir_exists("build")
        self.config.write_config("build/build_config.cfg")

        cmake_args = []
        storm_dir = self.config.get_as_string("storm_dir")
        if storm_dir:
            cmake_args += ['-Dstorm_DIR=' + storm_dir]
        _ = subprocess.check_output(['cmake', os.path.abspath("cmake")] + cmake_args, cwd=build_temp_version)
        cmake_conf = setup_helper.load_cmake_config(os.path.join(build_temp_version, 'generated/config.py'))

        # Set storm directory
        if storm_dir == "":
            storm_dir = cmake_conf.STORM_DIR
        if storm_dir != cmake_conf.STORM_DIR:
            print("Stormpy - Warning: Using different storm directory {} instead of given {}!".format(
                cmake_conf.STORM_DIR,
                storm_dir))
            storm_dir = cmake_conf.STORM_DIR

        # Check version
        storm_version, storm_commit = setup_helper.parse_storm_version(cmake_conf.STORM_VERSION)
        if StrictVersion(storm_version) < StrictVersion(storm_min_version):
            sys.exit(
                'Stormpy - Error: Storm version {} from \'{}\' is not supported anymore!'.format(storm_version,
                                                                                                 storm_dir))

        # Check additional support
        use_dft = cmake_conf.HAVE_STORM_DFT and not self.config.get_as_bool("disable_dft")
        use_pars = cmake_conf.HAVE_STORM_PARS and not self.config.get_as_bool("disable_pars")

        # Print build info
        print("Stormpy - Using storm {} from {}".format(storm_version, storm_dir))
        if use_dft:
            print("Stormpy - Support for DFTs found and included.")
        else:
            print("Stormpy - Warning: No support for DFTs!")
        if use_pars:
            print("Stormpy - Support for parametric models found and included.")
        else:
            print("Stormpy - Warning: No support for parametric models!")

        # Set general cmake build options
        build_type = 'Debug' if self.config.get_as_bool("debug") else 'Release'
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable]
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + build_type]
        if storm_dir is not None:
            cmake_args += ['-Dstorm_DIR=' + storm_dir]
        if use_dft:
            cmake_args += ['-DHAVE_STORM_DFT=ON']
        if use_pars:
            cmake_args += ['-DHAVE_STORM_PARS=ON']
        build_args = ['--config', build_type]
        build_args += ['--', '-j{}'.format(self.config.get_as_int("jobs"))]

        # Build extensions
        for ext in self.extensions:
            setup_helper.ensure_dir_exists(os.path.join(self._extdir(ext.name), ext.subdir))
            if ext.name == "core":
                with open(os.path.join(self._extdir(ext.name), ext.subdir, "_config.py"), "w") as f:
                    f.write("# Generated from setup.py at {}\n".format(datetime.datetime.now()))

                    f.write("import pycarl\n")
                    if cmake_conf.STORM_CLN_EA or cmake_conf.STORM_CLN_RF:
                        f.write("import pycarl.cln\n")
                    if not cmake_conf.STORM_CLN_EA or not cmake_conf.STORM_CLN_RF:
                        f.write("import pycarl.gmp\n")

                    if cmake_conf.STORM_CLN_EA:
                        f.write("Rational = pycarl.cln.Rational\n")
                    else:
                        f.write("Rational = pycarl.gmp.Rational\n")

                    if cmake_conf.STORM_CLN_RF:
                        rfpackage = "cln"
                    else:
                        rfpackage = "gmp"
                    f.write("RationalRF = pycarl.{}.Rational\n".format(rfpackage))
                    f.write("Polynomial = pycarl.{}.Polynomial\n".format(rfpackage))
                    f.write("FactorizedPolynomial = pycarl.{}.FactorizedPolynomial\n".format(rfpackage))
                    f.write("RationalFunction = pycarl.{}.RationalFunction\n".format(rfpackage))
                    f.write("FactorizedRationalFunction = pycarl.{}.FactorizedRationalFunction\n".format(rfpackage))
                    f.write("\n")
                    f.write("storm_with_dft = {}\n".format(use_dft))
                    f.write("storm_with_pars = {}\n".format(use_pars))

            elif ext.name == "info":
                with open(os.path.join(self._extdir(ext.name), ext.subdir, "_config.py"), "w") as f:
                    f.write("# Generated from setup.py at {}\n".format(datetime.datetime.now()))
                    f.write("storm_version = \"{}\"\n".format(storm_version))
                    f.write("storm_cln_ea = {}\n".format(cmake_conf.STORM_CLN_EA))
                    f.write("storm_cln_rf = {}".format(cmake_conf.STORM_CLN_RF))
            elif ext.name == "dft":
                with open(os.path.join(self._extdir(ext.name), ext.subdir, "_config.py"), "w") as f:
                    f.write("# Generated from setup.py at {}\n".format(datetime.datetime.now()))
                    f.write("storm_with_dft = {}".format(use_dft))
                if not use_dft:
                    print("Stormpy - DFT bindings skipped")
                    continue
            elif ext.name == "pars":
                with open(os.path.join(self._extdir(ext.name), ext.subdir, "_config.py"), "w") as f:
                    f.write("# Generated from setup.py at {}\n".format(datetime.datetime.now()))
                    f.write("storm_with_pars = {}".format(use_pars))
                if not use_pars:
                    print("Stormpy - Bindings for parametric models skipped")
                    continue
            self.build_extension(ext, cmake_args, build_args)

    def initialize_options(self):
        build_ext.initialize_options(self)
        # Load setup config
        self.config.load_from_file("build/build_config.cfg")
        # Set default values for custom cmdline flags
        self.storm_dir = None
        self.disable_dft = None
        self.disable_pars = None
        self.debug = None
        self.jobs = None

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Update setup config
        self.config.update("storm_dir", self.storm_dir)
        self.config.update("disable_dft", self.disable_dft)
        self.config.update("disable_pars", self.disable_pars)
        self.config.update("debug", self.debug)
        self.config.update("jobs", self.jobs)

    def build_extension(self, ext, general_cmake_args, general_build_args):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.join(extdir, ext.subdir)] + general_cmake_args
        build_args = general_build_args

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        setup_helper.ensure_dir_exists(self.build_temp)
        print("Pycarl - CMake args={}".format(cmake_args))
        # Call cmake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', ext.name] + build_args, cwd=self.build_temp)


class PyTest(test):
    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['tests'])
        sys.exit(errno)


setup(
    name="stormpy",
    version=setup_helper.obtain_version(),
    author="M. Volk",
    author_email="matthias.volk@cs.rwth-aachen.de",
    maintainer="S. Junges",
    maintainer_email="sebastian.junges@cs.rwth-aachen.de",
    url="http://moves.rwth-aachen.de",
    description="stormpy - Python Bindings for Storm",
    long_description='',
    packages=['stormpy', 'stormpy.info', 'stormpy.logic', 'stormpy.storage', 'stormpy.utility',
              'stormpy.pars', 'stormpy.dft'],
    package_dir={'': 'lib'},
    ext_package='stormpy',
    ext_modules=[CMakeExtension('core', subdir=''),
                 CMakeExtension('info', subdir='info'),
                 CMakeExtension('logic', subdir='logic'),
                 CMakeExtension('storage', subdir='storage'),
                 CMakeExtension('utility', subdir='utility'),
                 CMakeExtension('dft', subdir='dft'),
                 CMakeExtension('pars', subdir='pars'),
                 ],
    cmdclass={'build_ext': CMakeBuild, 'test': PyTest},
    zip_safe=False,
    install_requires=['pycarl>=2.0.1'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3',
)

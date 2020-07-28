from spack import *

class Magmadnn(CMakePackage):
    """
    High Performance Deep Learning Package
    """

    homepage = 'https://github.com/MagmaDNN/magmadnn'
    url = 'https://github.com/MagmaDNN/magmadnn/archive/v1.0.tar.gz'
    git = 'https://github.com/MagmaDNN/magmadnn.git'

    maintainers = ['Dando18', 'flipflapflop']

    version('1.0', commit='cd7db2727a08dbe25875875fe9086b780577596e')
    version('develop', branch='dev')

    variant('cuda', default=False, description='Build with CUDA support')
    variant('mpi', default=False, description='Build with MPI support')
    variant('openmp', default=False, description='Build with OpenMP support')
    variant('onednn', default=False, description='Build with OneDNN support')
    variant('shared', default=True, description='Build shared library')
    variant('docs', default=False, description='Build docs')

    depends_on('cmake@3.9:', type='build')
    depends_on('blas')
    depends_on('lapack')
    depends_on('cuda', when='+cuda')
    depends_on('cudnn', when='+cuda')
    depends_on('magma', when='+cuda')
    depends_on('onednn', when='+onednn')
    depends_on('mpi', when='+mpi')
    depends_on('openmp', when='+openmp')
    depends_on('doxygen', when='+docs')


    '''
    option(MAGMADNN_ENABLE_CUDA "Enable use of CUDA library and compilation of CUDA kernel" OFF)
    option(MAGMADNN_ENABLE_MPI "Enable distributed memory routines using MPI" OFF)
    option(MAGMADNN_ENABLE_OMP "Enable parallelization using OpenMP library" OFF)
    option(MAGMADNN_ENABLE_MKLDNN "Enable use of MKLDNN library" OFF)
    option(MAGMADNN_BUILD_MKLDNN "Enable build of MKLDNN from source" OFF)
    option(MAGMADNN_BUILD_DOC "Generate documentation" OFF)
    option(MAGMADNN_BUILD_EXAMPLES "Build MagmaDNN examples" ON)
    option(MAGMADNN_BUILD_TESTS "Generate build files for unit tests" OFF)
    option(MAGMADNN_BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)
    '''
    
    def cmake_args(self):
        spec = self.spec
        return [
            self.define_from_variant('MAGMADNN_ENABLE_CUDA', 'cuda'),
            self.define_from_variant('MAGMADNN_ENABLE_MPI', 'mpi'),
            self.define_from_variant('MAGMADNN_ENABLE_OMP', 'openmp'),
            self.define_from_variant('MAGMADNN_ENABLE_MKLDNN', 'onednn'),
            self.define_from_variant('MAGMADNN_BUILD_SHARED_LIBS', 'shared'),
            self.define_from_variant('MAGMADNN_BUILD_DOC', 'docs')
        ]
    

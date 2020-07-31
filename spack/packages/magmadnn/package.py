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
    depends_on('doxygen', when='+docs')


    def cmake_args(self):
        return [
            self.define_from_variant('MAGMADNN_ENABLE_CUDA', 'cuda'),
            self.define_from_variant('MAGMADNN_ENABLE_MPI', 'mpi'),
            self.define_from_variant('MAGMADNN_ENABLE_OMP', 'openmp'),
            self.define_from_variant('MAGMADNN_ENABLE_MKLDNN', 'onednn'),
            self.define_from_variant('MAGMADNN_BUILD_SHARED_LIBS', 'shared'),
            self.define_from_variant('MAGMADNN_BUILD_DOC', 'docs')
        ]
    

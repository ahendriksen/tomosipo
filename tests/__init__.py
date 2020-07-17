import pytest
import astra

cuda_available = astra.use_cuda()
skip_if_no_cuda = pytest.mark.skipif(not cuda_available, reason="Cuda not available")

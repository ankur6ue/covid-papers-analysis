import os
import torch
cache = {}
use_cuda = True

base_path = os.environ.get('BASE_PATH')
# model paths
bert_large_path = os.path.join(base_path, 'data/models/bert-large-uncased-whole-word-masking-finetuned-squad')
bert_small_path = os.path.join(base_path, 'data/models/bert-base-uncased_finetuned_squad')
cord19q_path = os.path.join(base_path, 'data/cord19q/current')

# GPU settings
print('USE_GPU: {0}'.format(os.environ.get('USE_GPU')))
if os.environ.get('USE_GPU') == '1':
    use_cuda = True
    print('using CUDA')
else:
    use_cuda = False

num_gpus = torch.cuda.device_count()


# Global state that records if GPU is currently processing a request
class GpuState:
    def __init__(self):
        self._busy = False

    @property
    def busy(self):
        return self._busy

    @busy.setter
    def busy(self, a):
        self._busy = a


class CurrModel:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        # print("getter method called")
        return self._model

    @model.setter
    def model(self, a):
        # print("setter method called")
        self._model = a


gpu_state = GpuState()
# currently active model
curr_model = CurrModel()


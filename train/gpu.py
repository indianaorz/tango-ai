#test that cuda works
import torch

def test_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_initialized())
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())

test_cuda()
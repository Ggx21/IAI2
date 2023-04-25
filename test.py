import torch
print("cuda_version:",torch.version.cuda)
print("torch_version:",torch.__version__)
print('torch.cuda.current_device():',torch.cuda.current_device())
print('torch.cuda.device(0):',torch.cuda.device(0))
print('torch.cuda.device_count():',torch.cuda.device_count())
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0))
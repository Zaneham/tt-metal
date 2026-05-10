import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal

device_id = 0
device = ttnn.open_device(device_id=device_id)

# Create input tensor with randn and convert to int32
torch_a = torch.randn((1, 128 * 1024), dtype=torch.float32)
torch_a = (torch_a * 100).to(torch.int32)

# Compute expected output using PyTorch with dimension
torch_output, _ = torch.max(torch_a, dim=1, keepdim=True)

# Convert to ttnn and run max with dimension
a = ttnn.from_torch(torch_a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
output = ttnn.max(a, dim=1, keepdim=True)

# Convert back and compare
output_torch = ttnn.to_torch(output)

print(output_torch)
print(torch_output)

assert_equal(output_torch, torch_output)
print(f"max(dim=1, keepdim=True): PASS")

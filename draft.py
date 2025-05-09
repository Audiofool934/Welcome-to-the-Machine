import torch

x = torch.tensor([1.,2.,3.])
w = torch.tensor([0.1,0.2,0.3], requires_grad=True)

# For Jacobian
# Define the function whose Jacobian is sought
def compute_model_output(weights_arg):
  # 'x' is captured from the outer scope where it's defined.
  # 'weights_arg' will be the 'w' tensor when 'jacobian' is called.
  return torch.stack([weights_arg @ x, (weights_arg**2).sum()])

jacobian = torch.autograd.functional.jacobian

print(jacobian(compute_model_output, w))
#include <torch/extension.h>

#include <ipex.h>

#include <vector>

template <typename scalar_t>
scalar_t sigmoid(scalar_t z) {
  return 1.0f / (1.0f + exp(-z));
}

template <typename scalar_t>
scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0f - s) * s;
}

template <typename scalar_t>
scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1.0f - (t * t);
}

template <typename scalar_t>
scalar_t elu(scalar_t z, scalar_t alpha = 1.0f) {
  return fmax(0.0f, z) + fmin(0.0f, alpha * (exp(z) - 1.0f));
}

template <typename scalar_t>
scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0f) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0f ? 0.0f : 1.0f;
  return d_relu + (((alpha * (e - 1.0f)) < 0.0f) ? (alpha * e) : 0.0f);
}

template <typename scalar_t>
void lltm_xpu_forward_kernel(
        const scalar_t* gates,
        const scalar_t* old_cell,
        scalar_t* new_h,
        scalar_t* new_cell,
        scalar_t* input_gate,
        scalar_t* output_gate,
        scalar_t* candidate_cell,
        size_t state_size,
        size_t batch_size) {

  const int threads = 1024;
  const int work_groups = (state_size + threads - 1) / threads;

  // define the kernel
  auto cgf = [&](sycl::handler& cgh) {
    auto kfn = [=](sycl::nd_item<2> item) {

      const int column = item.get_group(0) * item.get_group_range(0) + item.get_local_id(0);
      const int index = item.get_group(1) * state_size + column;
      const int gates_row = item.get_group(1) * (state_size * 3);

      if (column < state_size) {
        input_gate[index] = sigmoid(gates[gates_row + column]);
        output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
        candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
        new_cell[index] =
                old_cell[index] + candidate_cell[index] * input_gate[index];
        new_h[index] = tanh(new_cell[index]) * output_gate[index];
      }

    };

    cgh.parallel_for(
            sycl::nd_range<2>(
                    sycl::range<2>(work_groups * threads, batch_size),
                    sycl::range<2>(threads, 1)),
            kfn);
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);

  queue.submit(cgf);
}

std::vector<torch::Tensor> lltm_xpu_forward(
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor bias,
        torch::Tensor old_h,
        torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gates = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_xpu", ([&] {
    lltm_xpu_forward_kernel<scalar_t>(
          gates.data<scalar_t>(),
                  old_cell.data<scalar_t>(),
                  new_h.data<scalar_t>(),
                  new_cell.data<scalar_t>(),
                  input_gate.data<scalar_t>(),
                  output_gate.data<scalar_t>(),
                  candidate_cell.data<scalar_t>(),
                  state_size,
                  batch_size);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

template <typename scalar_t>
void lltm_xpu_backward_kernel(
        torch::PackedTensorAccessor32<scalar_t,2> d_old_cell,
        torch::PackedTensorAccessor32<scalar_t,3> d_gates,
        const torch::PackedTensorAccessor32<scalar_t,2> grad_h,
        const torch::PackedTensorAccessor32<scalar_t,2> grad_cell,
        const torch::PackedTensorAccessor32<scalar_t,2> new_cell,
        const torch::PackedTensorAccessor32<scalar_t,2> input_gate,
        const torch::PackedTensorAccessor32<scalar_t,2> output_gate,
        const torch::PackedTensorAccessor32<scalar_t,2> candidate_cell,
        const torch::PackedTensorAccessor32<scalar_t,3> gate_weights,
        size_t state_size,
        size_t batch_size) {

  const int threads = 1024;
  const int work_groups = (state_size + threads - 1) / threads;

  // define the kernel
  auto cgf = [&](sycl::handler& cgh) {
    auto kfn = [=](sycl::nd_item<2> item) {
      //batch index
      const int n = item.get_group(1);
      // column index
      const int c = item.get_group(0) * item.get_group_range(0) + item.get_local_id(0);
      auto d_gates_ = d_gates;
      auto d_old_cell_ = d_old_cell;
      if (c < d_gates.size(2)){
        const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
        const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
        const auto d_new_cell =
                d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


        d_old_cell_[n][c] = d_new_cell;
        const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
        const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

        d_gates_[n][0][c] =
                d_input_gate * d_sigmoid(gate_weights[n][0][c]);
        d_gates_[n][1][c] =
                d_output_gate * d_sigmoid(gate_weights[n][1][c]);
        d_gates_[n][2][c] =
                d_candidate_cell * d_elu(gate_weights[n][2][c]);
      }
    };

    cgh.parallel_for(
            sycl::nd_range<2>(
                    sycl::range<2>(work_groups * threads, batch_size),
                    sycl::range<2>(threads, 1)),
            kfn);
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);

  queue.submit(cgf);
}

std::vector<torch::Tensor> lltm_xpu_backward(
        torch::Tensor grad_h,
        torch::Tensor grad_cell,
        torch::Tensor new_cell,
        torch::Tensor input_gate,
        torch::Tensor output_gate,
        torch::Tensor candidate_cell,
        torch::Tensor X,
        torch::Tensor gates,
        torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_xpu", ([&] {
    lltm_xpu_backward_kernel<scalar_t>(
          d_old_cell.packed_accessor32<scalar_t,2>(),
                  d_gates.packed_accessor32<scalar_t,3>(),
                  grad_h.packed_accessor32<scalar_t,2>(),
                  grad_cell.packed_accessor32<scalar_t,2>(),
                  new_cell.packed_accessor32<scalar_t,2>(),
                  input_gate.packed_accessor32<scalar_t,2>(),
                  output_gate.packed_accessor32<scalar_t,2>(),
                  candidate_cell.packed_accessor32<scalar_t,2>(),
                  gates.packed_accessor32<scalar_t,3>(),
                  state_size,
                  batch_size);
  }));

  auto d_gate_weights = d_gates.reshape({batch_size, 3*state_size});
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
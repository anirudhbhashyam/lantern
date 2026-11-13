import tensor from "./lib/tensor";
fn main() {
    let a = tensor::from_data<f32>([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = tensor::matmul(tensor::view(a), tensor::transpose(tensor::view(a), 0, 1));
    print(b.data);
}

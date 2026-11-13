import tensor from "./lib/tensor";
import tensor.nn;
import std;
import math;

fn check_close(a: f32, b: f32, tol: f32 = 0.0001) {
    if abs(a - b) > tol || !is_finite(a) {
        print("got ", a, ", expected ", b);
        abort("values differ");
    }
}

fn check_data(a: const f32[:], b: const f32[:]) {
    assert(a.len == b.len);
    for i in a.len { check_close(a[i], b[i]); }
}

fn test_views() {
    var a = tensor::from_data<i64>([2, 3], [1, 2, 3, 4, 5, 6]);
    let v = tensor::view(a);
    assert(tensor::at(v, [0, 2]) == 3);
    let tr = tensor::transpose(v, 0, 1);
    assert(!tensor::is_contiguous(tr));
    tensor::set(tr, [2, 1], 7);
    assert(a.data[5] == 7);
    let step = tensor::slice(v, 1, 0, 3, 2);
    let packed = tensor::copy(step);
    assert(packed.data == [1, 3, 4, 7]);
    let r = tensor::reshape(tensor::view(packed), [4]);
    assert(tensor::at(r, [3]) == 7);
    let scalar = tensor::full<i64>([], 10);
    let result = tensor::add(v, tensor::view(scalar));
    assert(result.data == [11, 12, 13, 14, 15, 17]);
    let row = tensor::from_data<i64>([3], [1, 10, 100]);
    let broad = tensor::mul(v, tensor::view(row));
    assert(broad.data == [1, 20, 300, 4, 50, 700]);
    let empty = tensor::zeros<i64>([0, 3]);
    let e = tensor::add(tensor::view(empty), tensor::view(row));
    assert(e.data.len == 0 && e.shape[0] == 0 && e.shape[1] == 3);
    let axes = tensor::permute(v, [1, 0]);
    assert(tensor::at(axes, [2, 1]) == 7);
    let sum = tensor::sum_dim(tensor::view(empty), 0);
    assert(sum.data == [0, 0, 0]);
    let rank8 = tensor::ones<i64>([1, 1, 1, 1, 1, 1, 1, 2]);
    assert(tensor::sum_all(tensor::view(rank8)) == 2);
}

fn test_reductions() {
    let a = tensor::from_data<f32>([2, 3], [1000.0, 999.0, 998.0, -5.0, -10.0, -15.0]);
    let v = tensor::view(a);
    let mx = tensor::max_dim(v, 1);
    check_data(mx.data, [1000.0, -5.0]);
    let s = tensor::softmax(v, 1);
    let sums = tensor::sum_dim(tensor::view(s), 1);
    check_close(s.data[0], 0.66524094);
    check_close(sums.data[0], 1.0);
    check_close(sums.data[1], 1.0);
    check_close(tensor::mean_all(v), 494.5);
    let means = tensor::mean_dim(v, 1);
    check_data(means.data, [999.0, -10.0]);
}

fn test_gemm() {
    let a = tensor::from_data<f32>([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let av = tensor::view(a);
    let y = tensor::matmul(av, tensor::transpose(av, 0, 1));
    check_data(y.data, [14.0, 32.0, 32.0, 77.0]);
    let lhs = tensor::slice(av, 1, 0, 3, 2);
    let rhs = tensor::transpose(lhs, 0, 1);
    let z = tensor::matmul(lhs, rhs);
    check_data(z.data, [10.0, 22.0, 22.0, 52.0]);
    let padded = tensor::slice(av, 1, 1, 3);
    let q = tensor::matmul(padded, tensor::transpose(padded, 0, 1));
    check_data(q.data, [13.0, 28.0, 28.0, 61.0]);
    let e = tensor::zeros<f32>([2, 0]);
    let f = tensor::zeros<f32>([0, 2]);
    let g = tensor::matmul(tensor::view(e), tensor::view(f));
    check_data(g.data, [0.0, 0.0, 0.0, 0.0]);
    let batch = tensor::from_data<f32>([2, 1, 1, 2], [2.0, 3.0, 4.0, 5.0]);
    let rhs_batch = tensor::from_data<f32>([1, 3, 2, 1], [1.0, 0.0, 0.0, 1.0, 2.0, -1.0]);
    let result = tensor::batched_matmul(tensor::view(batch), tensor::view(rhs_batch));
    assert(result.rank == 4 && result.shape[..4] == [2, 3, 1, 1]);
    check_data(result.data, [2.0, 3.0, 1.0, 4.0, 5.0, 3.0]);
    let ints = tensor::from_data<i64>([1, 2], [3, 4]);
    let iv = tensor::view(ints);
    let exact = tensor::matmul_generic(iv, tensor::transpose(iv, 0, 1));
    assert(exact.data[0] == 25);
}

fn loss_for(x: tensor::View<f32>, p: f32[:], target: const f32[:],
            y: tensor::View<f32>, dy: f32[:]) -> f32 {
    tensor::linear_forward(x, p, y, false);
    return tensor::mse_backward(y.data, target, dy);
}

fn test_gradients() {
    var x: f32[6] = [0.2, -0.8, 0.4, 1.1, 0.3, -1.3];
    var p: f32[8] = [0.1, -0.2, 0.5, 0.3, -0.1, 0.7, 0.1, 0.2];
    let target: f32[4] = [-0.9, 0.3, 0.6, -0.2];
    var y:  f32[4] = [0.0; 4];
    var dy: f32[4] = [0.0; 4];
    var dp: f32[8] = [0.0; 8];
    var dx: f32[6] = [0.0; 6];
    let xv = tensor::matrix(x, 2, 3);
    let yv = tensor::matrix(y, 2, 2);
    let base = loss_for(xv, p, target, yv, dy);
    tensor::linear_backward(xv, p, tensor::matrix(dy, 2, 2), dp, tensor::matrix(dx, 2, 3), true);
    let delta: f32 = 0.001;
    for i in p.len {
        let orig = p[i];
        p[i] = orig + delta;
        let hi = loss_for(xv, p, target, yv, dy);
        p[i] = orig - delta;
        let lo = loss_for(xv, p, target, yv, dy);
        p[i] = orig;
        check_close(dp[i], (hi - lo) / (2.0 * delta), 0.0002);
    }
    for i in x.len {
        let orig = x[i];
        x[i] = orig + delta;
        let hi = loss_for(xv, p, target, yv, dy);
        x[i] = orig - delta;
        let lo = loss_for(xv, p, target, yv, dy);
        x[i] = orig;
        check_close(dx[i], (hi - lo) / (2.0 * delta), 0.0002);
    }
    var m: f32[8] = [0.0; 8];
    var v: f32[8] = [0.0; 8];
    var ema = p;
    tensor::adam_step(p, dp, m, v, ema, 1, 0.01, 0.5);
    assert(loss_for(xv, p, target, yv, dy) < base);
    assert(ema[0] != p[0] && ema[0] != 0.1);
}

fn main() {
    test_views();
    test_reductions();
    test_gemm();
    test_gradients();
    print("Tensor tests passed (views, broadcasting, reductions, GEMM, gradients, Adam).");
}

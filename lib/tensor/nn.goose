namespace tensor;
import .tensor;
import math;
import std;

// Borrow a row-major matrix from a caller-owned workspace.
fn matrix(data: f32[:], rows: i64, cols: i64) -> View<f32> {
    let s = shape_of([rows, cols]);
    assert(numel(s) == data.len);
    View<f32> { rank: 2, shape: s, stride: contiguous_strides(s, 2), offset: 0, data: data }
}

fn randn_into(out: f32[:], r: rng&) {
    var i = 0;
    while i < out.len {
        let u = 2.0 * rand_f32(r) - 1.0;
        let v = 2.0 * rand_f32(r) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            let factor = sqrt(-2.0 * log(s) / s);
            out[i] = u * factor;
            if i + 1 < out.len { out[i + 1] = v * factor; }
            i += 2;
        }
    }
}

fn randn(dims: const i64[:], r: rng&) -> Array<f32> {
    var out = zeros<f32>(dims);
    randn_into(out.data[..], r);
    return out;
}

// Dense weights are [fan_in, fan_out] followed by a bias vector.
fn linear_init(p: f32[:], fan_in: i64, fan_out: i64, r: rng&, gain: f32 = 1.0) {
    let n = fan_in * fan_out;
    assert(p.len == n + fan_out);
    let radius = gain * sqrt(6.0 / ((fan_in + fan_out) as f32));
    for i in n { p[i] = (2.0 * rand_f32(r) - 1.0) * radius; }
    for i in fan_out { p[n + i] = 0; }
}

// This training API reuses caller-owned output and gradient buffers.
// All input, parameter, output, and gradient spans must be disjoint.
fn linear_forward(x: View<f32>, p: f32[:], y: View<f32>, activate: bool) {
    let ni = x.shape[1];
    let no = y.shape[1];
    let n = ni * no;
    assert(p.len == n + no);
    assert(y.offset == 0 && y.data.len == numel(y.shape) && is_contiguous(y));
    gemm_into(x, matrix(p[..n], ni, no), y);
    for row in y.shape[0] {
        let start = row * no;
        for j in no {
            let z = y.data[start + j] + p[n + j];
            y.data[start + j] = if activate && z < 0.0 { 0.0 } else { z };
        }
    }
}

// dy already includes the output activation derivative (if any).
fn linear_backward(x: View<f32>, p: f32[:], dy: View<f32>, dp: f32[:], dx: View<f32>, input_grad: bool) {
    let ni = x.shape[1];
    let no = dy.shape[1];
    let n = ni * no;
    assert(p.len == n + no && dp.len == p.len);
    assert(dy.offset == 0 && dy.data.len == numel(dy.shape) && is_contiguous(dy));
    gemm_into(transpose(x, 0, 1), dy, matrix(dp[..n], ni, no));
    for j in no { dp[n + j] = 0; }
    for row in dy.shape[0] {
        let start = row * no;
        for j in no { dp[n + j] += dy.data[start + j]; }
    }
    if input_grad { gemm_into(dy, transpose(matrix(p[..n], ni, no), 0, 1), dx); }
}

fn relu_backward_into(dx: f32[:], activated: const f32[:]) {
    assert(dx.len == activated.len);
    for i in dx.len { if activated[i] <= 0.0 { dx[i] = 0; } }
}

fn mse_backward(pred: const f32[:], target: const f32[:], grad: f32[:]) -> f32 {
    assert(pred.len > 0 && pred.len == target.len && pred.len == grad.len);
    let scale: f32 = 2.0 / pred.len;
    var loss: f32 = 0;
    for i in pred.len {
        let error = pred[i] - target[i];
        grad[i] = scale * error;
        loss += error * error;
    }
    return loss / pred.len;
}

extern fn adam_kernel(p: f32[:], g: const f32[:], m: f32[:], v: f32[:], ema: f32[:],
                      rate: f32, eps: f32, decay: f32) = "goose_tensor_adam";

fn adam_step(p: f32[:], g: const f32[:], m: f32[:], v: f32[:], ema: f32[:],
             step: i64, lr: f32, decay: f32) {
    assert(step > 0 && p.len == g.len && p.len == m.len && p.len == v.len && p.len == ema.len);
    assert(lr > 0.0 && decay >= 0.0 && decay < 1.0);
    let c = sqrt(1.0 - pow(0.999 as f32, step as f32));
    let rate = lr * c / (1.0 - pow(0.9 as f32, step as f32));
    adam_kernel(p, g, m, v, ema, rate, 0.00000001 * c, decay);
}

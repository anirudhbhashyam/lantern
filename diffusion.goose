import tensor from "./lib/tensor";
import tensor.nn;
import std;
import math;
import os;
import binary;

const PIXELS = 784;
const CLASSES = 10;
const TIME_DIM = 32;
const INPUTS = PIXELS + CLASSES + TIME_DIM;
const WIDTH = 512;
const BATCH = 128;
const STEPS = 64;
const TRAIN_COUNT = 60000;
const P1 = (INPUTS + 1) * WIDTH;
const P2 = P1 + (WIDTH + 1) * WIDTH;
const P3 = P2 + (WIDTH + 1) * WIDTH;
const PARAMS = P3 + (WIDTH + 1) * PIXELS;
const CHECKPOINT = "artifacts/mnist-model.gdm";

fn make_schedule() -> f32[STEPS + 1] {
    var out: f32[STEPS + 1] = [0.0; STEPS + 1];
    let norm = cos(1.5707963267948966 * 0.008 / 1.008);
    for i in STEPS + 1 {
        let angle = 1.5707963267948966 * ((i as f64) / STEPS + 0.008) / 1.008;
        let c = cos(angle) / norm;
        out[i] = (c * c) as f32;
    }
    out[0] = 1.0;
    out[STEPS] = 0.0;
    return out;
}

fn condition(row: f32[:], digit: i64, step: i64) {
    for d in CLASSES { row[PIXELS + d] = if d == digit { 3.0 } else { 0.0 }; }
    for j in TIME_DIM / 2 {
        let omega = exp(-0.36 * (j as f32));
        let theta = step * omega;
        row[PIXELS + CLASSES + 2 * j] = cos(theta);
        row[PIXELS + CLASSES + 2 * j + 1] = sin(theta);
    }
}

fn network(x: tensor::View<f32>, p: f32[:], h1: tensor::View<f32>, h2: tensor::View<f32>,
           h3: tensor::View<f32>, y: tensor::View<f32>) {
    tensor::linear_forward(x, p[..P1], h1, true);
    tensor::linear_forward(h1, p[P1..P2], h2, true);
    tensor::linear_forward(h2, p[P2..P3], h3, true);
    tensor::linear_forward(h3, p[P3..], y, false);
}

fn save_model(p: const f32[:], updates: i64) {
    var bytes: u8[>..] = [];
    bytes.append("GDM2");
    binary::write_u32le(bytes, INPUTS);
    binary::write_u32le(bytes, WIDTH);
    binary::write_u32le(bytes, STEPS);
    binary::write_u64le(bytes, updates as u64);
    for i in p.len { binary::write_f32le(bytes, p[i]); }
    if !write_file_atomic(CHECKPOINT, bytes) { abort("could not write checkpoint (create artifacts/)"); }
}

fn train(updates: i64) {
    var images: u8[>..] = [];
    var labels: u8[>..] = [];
    if !read_file("data/train-images-idx3-ubyte", images)
        || !read_file("data/train-labels-idx1-ubyte", labels) {
        abort("MNIST IDX files missing; run ./run.sh data");
    }
    assert(images.len == 16 + TRAIN_COUNT * PIXELS && labels.len == 8 + TRAIN_COUNT);
    assert(binary::u32be(images, 0) == 2051 && binary::u32be(labels, 0) == 2049);
    assert(binary::u32be(images, 4) == TRAIN_COUNT && binary::u32be(labels, 4) == TRAIN_COUNT);
    assert(binary::u32be(images, 8) == 28 && binary::u32be(images, 12) == 28);

    var rng = rng { s: 20261113 };
    let schedule = make_schedule();
    var order: i64[>..] = [];
    order.resize(TRAIN_COUNT, 0);
    for i in TRAIN_COUNT { order[i] = i; }
    shuffle(order[..], rng);
    var cursor = 0;
    var p = tensor::zeros<f32>([PARAMS]);
    tensor::linear_init(p.data[..P1], INPUTS, WIDTH, rng, 1.4);
    tensor::linear_init(p.data[P1..P2], WIDTH, WIDTH, rng, 1.4);
    tensor::linear_init(p.data[P2..P3], WIDTH, WIDTH, rng, 1.4);
    tensor::linear_init(p.data[P3..], WIDTH, PIXELS, rng, 0.1);
    var ema = tensor::copy(tensor::view(p));
    var grad = tensor::zeros<f32>([PARAMS]);
    var m = tensor::zeros<f32>([PARAMS]);
    var v = tensor::zeros<f32>([PARAMS]);

    var x = tensor::zeros<f32>([BATCH, INPUTS]);
    var noise = tensor::zeros<f32>([BATCH, PIXELS]);
    var target = tensor::zeros<f32>([BATCH, PIXELS]);
    var h1 = tensor::zeros<f32>([BATCH, WIDTH]);
    var h2 = tensor::zeros<f32>([BATCH, WIDTH]);
    var h3 = tensor::zeros<f32>([BATCH, WIDTH]);
    var y = tensor::zeros<f32>([BATCH, PIXELS]);
    var d1 = tensor::zeros<f32>([BATCH, WIDTH]);
    var d2 = tensor::zeros<f32>([BATCH, WIDTH]);
    var d3 = tensor::zeros<f32>([BATCH, WIDTH]);
    var dy = tensor::zeros<f32>([BATCH, PIXELS]);

    let start = clock();
    var running: f32 = 0;
    for it in updates {
        tensor::randn_into(noise.data[..], rng);
        for row in BATCH {
            if cursor == TRAIN_COUNT {
                shuffle(order[..], rng);
                cursor = 0;
            }
            let i = order[cursor];
            cursor++;
            let digit = labels[8 + i] as i64;
            assert(digit < CLASSES);
            let step = 1 + rand_int(rng, STEPS);
            let signal = sqrt(schedule[step]);
            let sigma = sqrt(1.0 - schedule[step]);
            let ix = row * INPUTS;
            let iy = row * PIXELS;
            for j in PIXELS {
                let clean = (images[16 + i * PIXELS + j] as f32) * (2.0 / 255.0) - 1.0;
                let eps = noise.data[iy + j];
                x.data[ix + j] = signal * clean + sigma * eps;
                target.data[iy + j] = clean;
            }
            condition(x.data[ix..ix + INPUTS], digit, step);
        }

        let xv = tensor::view(x);
        let h1v = tensor::view(h1);
        let h2v = tensor::view(h2);
        let h3v = tensor::view(h3);
        network(xv, p.data[..], h1v, h2v, h3v, tensor::view(y));
        let loss = tensor::mse_backward(y.data[..], target.data[..], dy.data[..]);
        running += loss;
        assert(is_finite(loss));

        let d1v = tensor::view(d1);
        let d2v = tensor::view(d2);
        let d3v = tensor::view(d3);
        tensor::linear_backward(h3v, p.data[P3..], tensor::view(dy), grad.data[P3..], d3v, true);
        tensor::relu_backward_into(d3.data[..], h3.data[..]);
        tensor::linear_backward(h2v, p.data[P2..P3], d3v, grad.data[P2..P3], d2v, true);
        tensor::relu_backward_into(d2.data[..], h2.data[..]);
        tensor::linear_backward(h1v, p.data[P1..P2], d2v, grad.data[P1..P2], d1v, true);
        tensor::relu_backward_into(d1.data[..], h1.data[..]);
        tensor::linear_backward(xv, p.data[..P1], d1v, grad.data[..P1], xv, false);
        let progress = (it as f32) / updates;
        let lr:    f32 = (0.00025 + 0.00095 * 0.5 * (1.0 + cos(3.14159265 * progress))) * min<f32>(1.0, (it + 1) / 100.0);
        let decay: f32 = min<f32>(0.995, (it + 1.0) / (it + 10.0));
        tensor::adam_step(p.data[..], grad.data[..], m.data[..], v.data[..], ema.data[..], it + 1, lr, decay);

        if (it + 1) % 100 == 0 || it + 1 == updates {
            let count = 1 + it % 100;
            print("step ", it + 1, "/", updates, "  loss ", running / count, "  seconds ", clock() - start);
            running = 0;
        }
        if (it + 1) % 1000 == 0 || it + 1 == updates { save_model(ema.data[..], it + 1); }
    }
}

fn sample(seed: u64) {
    var bytes: u8[>..] = [];
    if !read_file(CHECKPOINT, bytes) { abort("checkpoint missing; train first"); }
    assert(bytes.len == 24 + 4 * PARAMS && bytes[..4] == "GDM2");
    assert(binary::u32le(bytes, 4) == INPUTS && binary::u32le(bytes, 8) == WIDTH
           && binary::u32le(bytes, 12) == STEPS);
    var p = tensor::zeros<f32>([PARAMS]);
    for i in PARAMS { p.data[i] = binary::f32_of_bits(binary::u32le(bytes, 24 + 4 * i)); }

    let schedule = make_schedule();
    var rng = rng { s: seed };
    var z = tensor::randn([CLASSES, PIXELS], rng);
    var x = tensor::zeros<f32>([CLASSES, INPUTS]);
    var h1 = tensor::zeros<f32>([CLASSES, WIDTH]);
    var h2 = tensor::zeros<f32>([CLASSES, WIDTH]);
    var h3 = tensor::zeros<f32>([CLASSES, WIDTH]);
    var y = tensor::zeros<f32>([CLASSES, PIXELS]);
    var frames: u8[>..] = [];
    var t = STEPS;
    loop {
        for i in z.data.len {
            assert(is_finite(z.data[i]));
            frames.push((127.5 * (1.0 + max(-1.0, min(1.0, z.data[i])))) as u8);
        }
        if t == 0 { break; }
        for d in CLASSES {
            let start = d * INPUTS;
            for j in PIXELS { x.data[start + j] = z.data[d * PIXELS + j]; }
            condition(x.data[start..start + INPUTS], d, t);
        }
        network(tensor::view(x), p.data[..], tensor::view(h1), tensor::view(h2), tensor::view(h3), tensor::view(y));
        let signal = sqrt(schedule[t]);
        let sigma = sqrt(1.0 - schedule[t]);
        let next_signal = sqrt(schedule[t - 1]);
        let next_sigma = sqrt(1.0 - schedule[t - 1]);
        for i in z.data.len {
            // Predict x0 directly and recover the noise for the DDIM update.
            let clean = clamp(y.data[i], -1.0, 1.0);
            let eps = (z.data[i] - signal * clean) / sigma;
            z.data[i] = next_signal * clean + next_sigma * eps;
        }
        t--;
    }
    if !write_file("artifacts/frames.gray", frames) { abort("could not write sample frames"); }
    print("wrote ", STEPS + 1, " reverse diffusion frames; checkpoint updates: ", binary::u64le(bytes, 16), "; seed: ", seed);
}

fn main() {
    let argv = args();
    if argv.len != 3 { abort("usage: diffusion train UPDATES | sample SEED"); }
    if argv[1] == "train" {
        let n, ok = parse_int(argv[2]);
        if !ok || n <= 0 { abort("UPDATES must be a positive integer"); }
        train(n);
    } else if argv[1] == "sample" {
        let seed, ok = parse_uint(argv[2]);
        if !ok { abort("SEED must be an unsigned integer"); }
        sample(seed);
    } else { abort("usage: diffusion train UPDATES | sample SEED"); }
}

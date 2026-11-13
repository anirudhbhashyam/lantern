namespace tensor;
import std;
import math;

const MAX_RANK = 8;

// Owned, contiguous row-major data. Borrowed views must not outlive the owner.
struct Array<T> { rank: i64, shape: i64[MAX_RANK], data: T[>..] }
struct View<T> { rank: i64, shape: i64[MAX_RANK], stride: i64[MAX_RANK], offset: i64, data: T[:] }

fn shape_of(dims: const i64[:]) -> i64[MAX_RANK] {
    assert(dims.len <= MAX_RANK);
    var s: i64[MAX_RANK] = [1; MAX_RANK];
    for d in dims.len {
        assert(dims[d] >= 0);
        s[d] = dims[d];
    }
    return s;
}

fn numel(shape: const i64[:]) -> i64 {
    var n = 1;
    for d in shape.len {
        assert(shape[d] >= 0);
        if shape[d] != 0 { assert(n <= 9223372036854775807 / shape[d]); }
        n *= shape[d];
    }
    return n;
}

fn full<T>(dims: const i64[:], value: T) -> Array<T> {
    let s = shape_of(dims);
    var out = Array<T> { rank: dims.len, shape: s, data: [] };
    out.data.resize(numel(s), value);
    return out;
}
fn zeros<T>(dims: const i64[:]) -> Array<T> { full(dims, default<T>()) }
fn ones<T>(dims: const i64[:]) -> Array<T> { full<T>(dims, 1) }
fn from_data<T>(dims: const i64[:], values: const T[:]) -> Array<T> {
    let s = shape_of(dims);
    assert(numel(s) == values.len);
    var out = Array<T> { rank: dims.len, shape: s, data: [] };
    out.data.append(values);
    return out;
}

fn contiguous_strides(shape: i64[MAX_RANK], rank: i64) -> i64[MAX_RANK] {
    var st: i64[MAX_RANK] = [0; MAX_RANK];
    var acc = 1;
    var d = rank;
    while d > 0 {
        d--;
        st[d] = acc;
        if shape[d] != 0 { assert(acc <= 9223372036854775807 / shape[d]); }
        acc *= shape[d];
    }
    return st;
}

fn view<T>(a: Array<T>&) -> View<T> {
    View<T> { rank: a.rank, shape: a.shape, stride: contiguous_strides(a.shape, a.rank),
              offset: 0, data: a.data[..] }
}

fn index_offset<T>(v: View<T>, idx: const i64[:]) -> i64 {
    assert(idx.len == v.rank);
    var off = v.offset;
    for d in idx.len {
        assert(0 <= idx[d] && idx[d] < v.shape[d]);
        off += idx[d] * v.stride[d];
    }
    return off;
}
fn at<T>(v: View<T>, idx: const i64[:]) -> T { v.data[index_offset(v, idx)] }
fn set<T>(v: View<T>, idx: const i64[:], value: T) {
    for d in v.rank { assert(v.stride[d] != 0 || v.shape[d] <= 1); }
    v.data[index_offset(v, idx)] = value;
}

fn flat_offset<T>(v: View<T>, flat: i64) -> i64 {
    var i = flat;
    var off = v.offset;
    var d = v.rank;
    while d > 0 {
        d--;
        off += i % v.shape[d] * v.stride[d];
        i /= v.shape[d];
    }
    return off;
}

fn transpose<T>(v: View<T>, a: i64, b: i64) -> View<T> {
    assert(0 <= a && a < v.rank && 0 <= b && b < v.rank);
    var out = v;
    out.shape[a] = v.shape[b];
    out.shape[b] = v.shape[a];
    out.stride[a] = v.stride[b];
    out.stride[b] = v.stride[a];
    return out;
}

fn slice<T>(v: View<T>, d: i64, lo: i64, hi: i64, step: i64 = 1) -> View<T> {
    assert(0 <= d && d < v.rank && step > 0);
    assert(0 <= lo && lo <= hi && hi <= v.shape[d]);
    var out = v;
    out.shape[d] = if lo == hi { 0 } else { 1 + (hi - lo - 1) / step };
    out.offset += lo * v.stride[d];
    if out.shape[d] > 1 {
        assert(v.stride[d] <= 9223372036854775807 / step);
        out.stride[d] *= step;
    }
    return out;
}

fn is_contiguous<T>(v: View<T>) -> bool {
    if numel(v.shape) == 0 { return true; }
    let st = contiguous_strides(v.shape, v.rank);
    for d in v.rank { if v.shape[d] > 1 && v.stride[d] != st[d] { return false; } }
    return true;
}

fn reshape<T>(v: View<T>, dims: const i64[:]) -> View<T> {
    let s = shape_of(dims);
    assert(is_contiguous(v) && numel(s) == numel(v.shape));
    View<T> { rank: dims.len, shape: s, stride: contiguous_strides(s, dims.len),
              offset: v.offset, data: v.data }
}

fn broadcast_to<T>(v: View<T>, dims: const i64[:]) -> View<T> {
    let s = shape_of(dims);
    assert(dims.len >= v.rank);
    var st: i64[MAX_RANK] = [0; MAX_RANK];
    let lead = dims.len - v.rank;
    for d in v.rank {
        if v.shape[d] == s[lead + d] { st[lead + d] = v.stride[d]; }
        else { assert(v.shape[d] == 1); }
    }
    View<T> { rank: dims.len, shape: s, stride: st, offset: v.offset, data: v.data }
}

fn copy<T>(v: View<T>) -> Array<T> {
    let n = numel(v.shape);
    if n > 0 && is_contiguous(v) {
        return from_data(v.shape[..v.rank], v.data[v.offset..v.offset + n]);
    }
    var out = zeros<T>(v.shape[..v.rank]);
    for i in n { out.data[i] = v.data[flat_offset(v, i)]; }
    return out;
}

fn map<T, F>(a: View<T>) -> Array<T> {
    var out = zeros<T>(a.shape[..a.rank]);
    if is_contiguous(a) {
        for i in out.data.len { out.data[i] = F(a.data[a.offset + i]); }
    } else {
        for i in out.data.len { out.data[i] = F(a.data[flat_offset(a, i)]); }
    }
    return out;
}

fn zip<T, F>(a: View<T>, b: View<T>) -> Array<T> {
    let rank = max(a.rank, b.rank);
    var dims: i64[MAX_RANK] = [1; MAX_RANK];
    for d in rank {
        let ai = d + a.rank - rank;
        let bi = d + b.rank - rank;
        let x = if ai < 0 { 1 } else { a.shape[ai] };
        let y = if bi < 0 { 1 } else { b.shape[bi] };
        assert(x == y || x == 1 || y == 1);
        dims[d] = if x == 1 { y } else { x };
    }
    let av = broadcast_to(a, dims[..rank]);
    let bv = broadcast_to(b, dims[..rank]);
    var out = zeros<T>(dims[..rank]);
    if is_contiguous(av) && is_contiguous(bv) {
        for i in out.data.len { out.data[i] = F(a.data[a.offset + i], b.data[b.offset + i]); }
    } else {
        for i in out.data.len { out.data[i] = F(a.data[flat_offset(av, i)], b.data[flat_offset(bv, i)]); }
    }
    return out;
}
fn add<T>(a: View<T>, b: View<T>) -> Array<T> { zip(a, b) { x, y => x + y } }
fn sub<T>(a: View<T>, b: View<T>) -> Array<T> { zip(a, b) { x, y => x - y } }
fn mul<T>(a: View<T>, b: View<T>) -> Array<T> { zip(a, b) { x, y => x * y } }
fn div<T>(a: View<T>, b: View<T>) -> Array<T> { zip(a, b) { x, y => x / y } }
fn relu<T>(a: View<T>) -> Array<T> { map(a) { x => if x > 0 { x } else { 0 } } }
fn exp_t<T>(a: View<T>) -> Array<T> { map(a) { x => exp(x) } }

fn reduce<T, F>(v: View<T>, d: i64, init: T) -> Array<T> {
    assert(0 <= d && d < v.rank);
    var base = v;
    base.shape[d] = 1;
    var out = full(base.shape[..v.rank], init);
    if v.shape[d] > 0 {
        for i in out.data.len {
            let off = flat_offset(base, i);
            var x = init;
            for k in v.shape[d] { x = F(x, v.data[off + k * v.stride[d]]); }
            out.data[i] = x;
        }
    }
    return out;
}
fn sum_dim<T>(v: View<T>, d: i64) -> Array<T> { reduce(v, d, default<T>()) { x, y => x + y } }
fn max_dim<T>(v: View<T>, d: i64) -> Array<T> {
    assert(0 <= d && d < v.rank && v.shape[d] > 0);
    var base = v;
    base.shape[d] = 1;
    var out = zeros<T>(base.shape[..v.rank]);
    for i in out.data.len {
        let off = flat_offset(base, i);
        var x = v.data[off];
        for k in v.shape[d] { x = max(x, v.data[off + k * v.stride[d]]); }
        out.data[i] = x;
    }
    return out;
}
fn sum_all<T>(v: View<T>) -> T {
    var x = default<T>();
    for i in numel(v.shape) { x += v.data[flat_offset(v, i)]; }
    return x;
}
fn mean_all<T>(v: View<T>) -> T {
    let n = numel(v.shape);
    assert(n > 0);
    return sum_all(v) / (n as T);
}
fn softmax<T>(v: View<T>, d: i64) -> Array<T> {
    let peak = max_dim(v, d);
    let shifted = sub(v, view(peak));
    let e = exp_t(view(shifted));
    let s = sum_dim(view(e), d);
    return div(view(e), view(s));
}

extern fn tensor_gemm(a: const f32[:], b: const f32[:], c: f32[:], m: i32, n: i32,
                      k: i32, lda: i32, ldb: i32, ta: i32, tb: i32) = "goose_tensor_gemm";

fn blas_layout(v: View<f32>) -> i64 {
    if v.stride[1] == 1 && v.stride[0] >= v.shape[1] && v.stride[0] <= 2147483647 { return 0; }
    if v.stride[0] == 1 && v.stride[1] >= v.shape[0] && v.stride[1] <= 2147483647 { return 1; }
    return -1;
}

// Row-major and transposed/padded matrices go straight to BLAS. Arbitrary
// views are packed as needed. The output owns fresh contiguous storage.
fn matmul(a: View<f32>, b: View<f32>) -> Array<f32> {
    assert(a.rank == 2 && b.rank == 2 && a.shape[1] == b.shape[0]);
    let ta = blas_layout(a);
    let tb = blas_layout(b);
    if ta < 0 {
        let pa = copy(a);
        if tb < 0 {
            let pb = copy(b);
            return matmul_dense(view(pa), view(pb));
        }
        return matmul_dense(view(pa), b);
    }
    if tb < 0 {
        let pb = copy(b);
        return matmul_dense(a, view(pb));
    }
    return matmul_dense(a, b);
}

fn matmul_dense(a: View<f32>, b: View<f32>) -> Array<f32> {
    var out = zeros<f32>([a.shape[0], b.shape[1]]);
    gemm_into(a, b, view(out));
    return out;
}

fn gemm_into(a: View<f32>, b: View<f32>, out: View<f32>) {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    assert(a.rank == 2 && b.rank == 2 && out.rank == 2 && k == b.shape[0]);
    assert(out.shape[0] == m && out.shape[1] == n && is_contiguous(out));
    if m == 0 || n == 0 { return; }
    let dest = out.data[out.offset..out.offset + m * n];
    if k == 0 {
        for i in dest.len { dest[i] = 0; }
        return;
    }
    let ta = blas_layout(a);
    let tb = blas_layout(b);
    assert(ta >= 0 && tb >= 0 && m <= 2147483647 && n <= 2147483647 && k <= 2147483647);
    let ae = a.offset + (m - 1) * a.stride[0] + (k - 1) * a.stride[1] + 1;
    let be = b.offset + (k - 1) * b.stride[0] + (n - 1) * b.stride[1] + 1;
    tensor_gemm(a.data[a.offset..ae], b.data[b.offset..be], dest,
                m as i32, n as i32, k as i32, a.stride[ta] as i32, b.stride[tb] as i32,
                ta as i32, tb as i32);
}

// Reorder every axis without copying.
fn permute<T>(v: View<T>, axes: const i64[:]) -> View<T> {
    assert(axes.len == v.rank);
    var seen: bool[MAX_RANK] = [false; MAX_RANK];
    var out = v;
    for d in v.rank {
        let from = axes[d];
        assert(0 <= from && from < v.rank && !seen[from]);
        seen[from] = true;
        out.shape[d] = v.shape[from];
        out.stride[d] = v.stride[from];
    }
    return out;
}

fn mean_dim<T>(v: View<T>, d: i64) -> Array<T> {
    assert(0 <= d && d < v.rank && v.shape[d] > 0);
    var out = sum_dim(v, d);
    let n = v.shape[d] as T;
    for i in out.data.len { out.data[i] /= n; }
    return out;
}

// Rank-2 fallback for numeric types other than f32 (and for exact integer work).
fn matmul_generic<T>(a: View<T>, b: View<T>) -> Array<T> {
    assert(a.rank == 2 && b.rank == 2 && a.shape[1] == b.shape[0]);
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    var out = zeros<T>([m, n]);
    for i in m {
        for p in k {
            let x = a.data[a.offset + i * a.stride[0] + p * a.stride[1]];
            let bp = b.offset + p * b.stride[0];
            for j in n { out.data[i * n + j] += x * b.data[bp + j * b.stride[1]]; }
        }
    }
    return out;
}

// Treat the last two dimensions as matrices and broadcast all leading axes.
fn batched_matmul(a: View<f32>, b: View<f32>) -> Array<f32> {
    assert(a.rank >= 2 && b.rank >= 2);
    let ra = a.rank - 2;
    let rb = b.rank - 2;
    let batch_rank = max(ra, rb);
    let m = a.shape[ra];
    let k = a.shape[ra + 1];
    let n = b.shape[rb + 1];
    assert(k == b.shape[rb]);
    var dims: i64[MAX_RANK] = [1; MAX_RANK];
    var da = dims;
    var db = dims;
    for d in batch_rank {
        let ai = d + ra - batch_rank;
        let bi = d + rb - batch_rank;
        let x = if ai >= 0 { a.shape[ai] } else { 1 };
        let y = if bi >= 0 { b.shape[bi] } else { 1 };
        assert(x == y || x == 1 || y == 1);
        dims[d] = if x == 1 { y } else { x };
        da[d] = dims[d];
        db[d] = dims[d];
    }
    dims[batch_rank] = m;
    dims[batch_rank + 1] = n;
    da[batch_rank] = m;
    da[batch_rank + 1] = k;
    db[batch_rank] = k;
    db[batch_rank + 1] = n;
    var out = zeros<f32>(dims[..batch_rank + 2]);
    if out.data.len == 0 || k == 0 { return out; }
    var av = broadcast_to(a, da[..batch_rank + 2]);
    var bv = broadcast_to(b, db[..batch_rank + 2]);
    // Collapse the matrix dimensions to find each batch's base offset.
    av.shape[batch_rank] = 1;
    av.shape[batch_rank + 1] = 1;
    bv.shape[batch_rank] = 1;
    bv.shape[batch_rank + 1] = 1;
    var left = View<f32> { rank: 2, shape: shape_of([m, k]), stride: [0; MAX_RANK], offset: 0, data: a.data };
    var right = View<f32> { rank: 2, shape: shape_of([k, n]), stride: [0; MAX_RANK], offset: 0, data: b.data };
    for d in 2 {
        left.stride[d] = a.stride[ra + d];
        right.stride[d] = b.stride[rb + d];
    }
    let size = m * n;
    for i in out.data.len / size {
        left.offset = flat_offset(av, i);
        right.offset = flat_offset(bv, i);
        if blas_layout(left) >= 0 && blas_layout(right) >= 0 {
            let dest = out.data[i * size..(i + 1) * size];
            let dv = View<f32> { rank: 2, shape: shape_of([m, n]),
                                 stride: contiguous_strides(shape_of([m, n]), 2), offset: 0, data: dest };
            gemm_into(left, right, dv);
        } else {
            let product = matmul(left, right);
            for j in size { out.data[i * size + j] = product.data[j]; }
        }
    }
    return out;
}

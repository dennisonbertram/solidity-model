// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Weights} from "./Weights.sol";

/// @notice A tiny (~3.8k param) RMSNorm + single-head causal self-attention
/// char-level language model, fully quantized (Q12 fixed point, int16
/// weights) and embedded as contract bytecode. Trained off-chain
/// (train/train.py) on a public-domain corpus; weights exported by
/// train/export.py into Weights.sol. Inference runs entirely on-chain.
///
/// PARAMS/EXP_LUT/VOCAB_BYTES are `bytes constant` (inline contract code).
/// Every external entrypoint copies them into memory once (`_tables()`) and
/// threads that copy through the whole forward pass — indexing a `bytes
/// constant` directly inside a hot loop re-expands memory on every access
/// under the optimizer's IR pipeline (quadratic gas blowup), so all reads
/// go through the in-memory copy instead.
contract OnChainLLM is Weights {
    event Generated(address indexed caller, uint256 seed, string prompt, string output);

    struct Tables {
        bytes params;
        bytes lut;
        bytes vocab;
    }

    function _tables() internal pure returns (Tables memory) {
        return Tables({params: PARAMS, lut: EXP_LUT, vocab: VOCAB_BYTES});
    }

    // ---- fixed-point primitives ----

    function _readInt16(Tables memory t, uint256 idx) internal pure returns (int256) {
        uint256 byteOff = idx * 2;
        uint16 raw = uint16(uint8(t.params[byteOff])) | (uint16(uint8(t.params[byteOff + 1])) << 8);
        return int256(int16(raw));
    }

    function _readLUT(Tables memory t, uint256 idx) internal pure returns (uint256) {
        uint256 byteOff = idx * 8;
        uint256 v;
        for (uint256 b = 0; b < 8; b++) {
            v |= uint256(uint8(t.lut[byteOff + b])) << (8 * b);
        }
        return v;
    }

    // Babylonian integer square root (standard, e.g. Uniswap V2 Math.sqrt).
    function _sqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }

    // Piecewise-linear lookup of exp(x) for x in Q(SCALE) fixed point,
    // returned scaled by LUT_SCALE (an arbitrary positive scale — softmax
    // ratios cancel it out, so precision is all that matters here).
    function _expFixed(Tables memory t, int256 xFixed) internal pure returns (uint256) {
        if (xFixed <= LUT_MIN_FIXED) xFixed = LUT_MIN_FIXED;
        if (xFixed >= LUT_MAX_FIXED) xFixed = LUT_MAX_FIXED;
        uint256 range = uint256(LUT_MAX_FIXED - LUT_MIN_FIXED);
        uint256 pos = uint256(xFixed - LUT_MIN_FIXED);
        uint256 numerator = pos * (LUT_N - 1);
        uint256 idx = numerator / range;
        if (idx > LUT_N - 2) idx = LUT_N - 2;
        uint256 fracNum = numerator - idx * range;
        uint256 lo = _readLUT(t, idx);
        uint256 hi = _readLUT(t, idx + 1);
        if (hi >= lo) {
            return lo + ((hi - lo) * fracNum) / range;
        }
        return lo - ((lo - hi) * fracNum) / range;
    }

    // ---- vector / matrix ops (all operate on Q(SCALE) fixed-point ints) ----

    function _embeddingRow(Tables memory t, uint256 tokenId) internal pure returns (int256[] memory out) {
        out = new int256[](C);
        for (uint256 i = 0; i < C; i++) {
            out[i] = _readInt16(t, OFF_TOK_EMB + tokenId * C + i);
        }
    }

    function _posEmbeddingRow(Tables memory t, uint256 pos) internal pure returns (int256[] memory out) {
        out = new int256[](C);
        for (uint256 i = 0; i < C; i++) {
            out[i] = _readInt16(t, OFF_POS_EMB + pos * C + i);
        }
    }

    function _add(int256[] memory a, int256[] memory b) internal pure returns (int256[] memory out) {
        out = new int256[](a.length);
        for (uint256 i = 0; i < a.length; i++) out[i] = a[i] + b[i];
    }

    function _relu(int256[] memory x) internal pure returns (int256[] memory out) {
        out = new int256[](x.length);
        for (uint256 i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : int256(0);
    }

    function _dotScaled(int256[] memory a, int256[] memory b) internal pure returns (int256) {
        int256 acc = 0;
        for (uint256 i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc / int256(SCALE);
    }

    function _rmsnorm(Tables memory t, int256[] memory x, uint256 gainOffset)
        internal
        pure
        returns (int256[] memory out)
    {
        uint256 n = x.length;
        uint256 sumSq = 0;
        for (uint256 i = 0; i < n; i++) {
            int256 xi = x[i];
            sumSq += uint256(xi * xi);
        }
        uint256 rms = _sqrt(sumSq / n);
        if (rms == 0) rms = 1;
        out = new int256[](n);
        for (uint256 i = 0; i < n; i++) {
            int256 normalized = (x[i] * int256(SCALE)) / int256(rms);
            int256 gain = _readInt16(t, gainOffset + i);
            out[i] = (normalized * gain) / int256(SCALE);
        }
    }

    // weightOffset holds an inDim x outDim matrix, row-major (row i = input i's weights).
    function _linear(
        Tables memory t,
        int256[] memory x,
        uint256 weightOffset,
        uint256 biasOffset,
        bool hasBias,
        uint256 outDim
    ) internal pure returns (int256[] memory out) {
        uint256 inDim = x.length;
        out = new int256[](outDim);
        for (uint256 j = 0; j < outDim; j++) {
            int256 acc = 0;
            for (uint256 i = 0; i < inDim; i++) {
                acc += x[i] * _readInt16(t, weightOffset + i * outDim + j);
            }
            acc = acc / int256(SCALE);
            if (hasBias) acc += _readInt16(t, biasOffset + j);
            out[j] = acc;
        }
    }

    function _softmaxWeights(Tables memory t, int256[] memory scores)
        internal
        pure
        returns (uint256[] memory weights, uint256 denom)
    {
        uint256 n = scores.length;
        weights = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            weights[i] = _expFixed(t, scores[i]);
            denom += weights[i];
        }
    }

    // ---- transformer block + head, run for the *last* context position only ----

    function _forward(Tables memory t, uint256[] memory ctx) internal pure returns (int256[] memory logits) {
        uint256 L = ctx.length;
        int256[][] memory xs = new int256[][](L);
        int256[][] memory xns = new int256[][](L);
        for (uint256 i = 0; i < L; i++) {
            xs[i] = _add(_embeddingRow(t, ctx[i]), _posEmbeddingRow(t, i));
            xns[i] = _rmsnorm(t, xs[i], OFF_ATTN_NORM_GAIN);
        }

        int256[] memory qLast = _linear(t, xns[L - 1], OFF_WQ, 0, false, C);
        int256[] memory scores = new int256[](L);
        int256[][] memory vAll = new int256[][](L);
        uint256 sqrtC = _sqrt(C);
        for (uint256 i = 0; i < L; i++) {
            int256[] memory kT = _linear(t, xns[i], OFF_WK, 0, false, C);
            vAll[i] = _linear(t, xns[i], OFF_WV, 0, false, C);
            scores[i] = _dotScaled(qLast, kT) / int256(sqrtC);
        }

        (uint256[] memory aw, uint256 aDenom) = _softmaxWeights(t, scores);
        int256[] memory attnOut = new int256[](C);
        for (uint256 c = 0; c < C; c++) {
            int256 acc = 0;
            for (uint256 i = 0; i < L; i++) {
                acc += int256(aw[i]) * vAll[i][c];
            }
            attnOut[c] = acc / int256(aDenom);
        }
        int256[] memory attnProj = _linear(t, attnOut, OFF_WO, 0, false, C);
        int256[] memory x2 = _add(xs[L - 1], attnProj);

        int256[] memory xn2 = _rmsnorm(t, x2, OFF_FFN_NORM_GAIN);
        int256[] memory h = _relu(_linear(t, xn2, OFF_W1, OFF_B1, true, HID));
        int256[] memory ff = _linear(t, h, OFF_W2, OFF_B2, true, C);
        int256[] memory x3 = _add(x2, ff);

        int256[] memory xn3 = _rmsnorm(t, x3, OFF_FINAL_NORM_GAIN);

        logits = new int256[](V);
        for (uint256 v = 0; v < V; v++) {
            logits[v] = _dotScaled(xn3, _embeddingRow(t, v));
        }
    }

    function _sampleToken(Tables memory t, int256[] memory logits, uint256 randomness)
        internal
        pure
        returns (uint256)
    {
        uint256 n = logits.length;
        uint256[] memory weights = new uint256[](n);
        uint256 denom = 0;
        for (uint256 i = 0; i < n; i++) {
            weights[i] = _expFixed(t, logits[i]);
            denom += weights[i];
        }
        uint256 threshold = randomness % denom;
        uint256 cum = 0;
        for (uint256 i = 0; i < n; i++) {
            cum += weights[i];
            if (threshold < cum) return i;
        }
        return n - 1;
    }

    function _charToId(Tables memory t, bytes1 c) internal pure returns (uint256) {
        for (uint256 i = 0; i < V; i++) {
            if (t.vocab[i] == c) return i;
        }
        revert("char not in vocab");
    }

    function _buildContext(Tables memory t, bytes calldata text) internal pure returns (uint256[] memory ctx) {
        uint256 ctxLen = text.length < TBLOCK ? text.length : TBLOCK;
        uint256 startIdx = text.length - ctxLen;
        ctx = new uint256[](ctxLen);
        for (uint256 i = 0; i < ctxLen; i++) {
            ctx[i] = _charToId(t, text[startIdx + i]);
        }
    }

    // ---- public interface ----

    function vocab() external pure returns (string memory) {
        return string(VOCAB_BYTES);
    }

    /// @notice Pure preview of next-token logits for a given context, no randomness/state.
    function nextTokenLogits(bytes calldata context) external pure returns (int256[] memory) {
        require(context.length >= 1, "empty context");
        Tables memory t = _tables();
        return _forward(t, _buildContext(t, context));
    }

    /// @notice Autoregressively generates `numTokens` characters continuing
    /// `prompt`, running the full forward pass on-chain for every token.
    /// Must be sent as a transaction (it emits an event, so it cannot be a
    /// view/pure call) — e.g. `cast send <addr> "generate(bytes,uint256,uint256)" ...`.
    // Each generated token re-runs the full forward pass (~6.9M gas measured
    // locally). Ethereum mainnet's 30M block gas limit fits ~4 tokens per
    // transaction; MAX_TOKENS is a generous local-testing ceiling, not a
    // mainnet-feasibility claim — for longer output, call `generate`
    // repeatedly, feeding the tail of the previous output back in as the
    // next prompt.
    uint256 internal constant MAX_TOKENS = 64;

    function generate(bytes calldata prompt, uint256 numTokens, uint256 seed)
        external
        returns (string memory output)
    {
        require(prompt.length >= 1, "empty prompt");
        require(numTokens >= 1 && numTokens <= MAX_TOKENS, "numTokens out of range");

        Tables memory t = _tables();
        uint256[] memory ctx = _buildContext(t, prompt);

        bytes32 bh = block.number > 0 ? blockhash(block.number - 1) : bytes32(0);
        bytes memory generated = new bytes(numTokens);
        for (uint256 step = 0; step < numTokens; step++) {
            int256[] memory logits = _forward(t, ctx);
            uint256 rnd = uint256(keccak256(abi.encodePacked(seed, bh, msg.sender, step)));
            uint256 nextId = _sampleToken(t, logits, rnd);
            generated[step] = VOCAB_BYTES[nextId];

            if (ctx.length < TBLOCK) {
                uint256[] memory grown = new uint256[](ctx.length + 1);
                for (uint256 i = 0; i < ctx.length; i++) grown[i] = ctx[i];
                grown[ctx.length] = nextId;
                ctx = grown;
            } else {
                uint256[] memory shifted = new uint256[](TBLOCK);
                for (uint256 i = 0; i < TBLOCK - 1; i++) shifted[i] = ctx[i + 1];
                shifted[TBLOCK - 1] = nextId;
                ctx = shifted;
            }
        }

        output = string(abi.encodePacked(prompt, generated));
        emit Generated(msg.sender, seed, string(prompt), output);
    }
}

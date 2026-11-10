# On-Chain LLM

A tiny language model whose weights live in Solidity bytecode and whose
forward pass runs entirely inside the EVM. Query it by sending a transaction.

**Key result:** about 3,800 parameters, 17.5KB of contract bytecode (under the
24KB limit), and about 6.9M gas per generated token, so roughly 4 tokens fit in
one 30M-gas block (see [Gas economics](#gas-economics)).

**Status:** working demo. It generates text on a local Anvil node and is covered
by Foundry tests. It has not been deployed to a public network, and the model
is too small to produce useful text.

## Contract design

`OnChainLLM.sol` implements a miniature Llama-style transformer:
token + position embedding → RMSNorm → single-head causal self-attention →
residual → RMSNorm → ReLU feed-forward → residual → RMSNorm → output logits
(tied to the token embedding). ~3,800 parameters, quantized to int16 fixed
point (Q12: `real = int16 / 4096`), trained with PyTorch on a public-domain
corpus (the Gettysburg Address), character-level over a 29-symbol vocab
(`a-z`, space, `.`, `,`).

All math (matmuls, RMSNorm, softmax) is plain integer fixed-point
arithmetic. `exp()` for softmax is a piecewise-linear lookup table (no
floating point exists in the EVM). Weights + LUT are ~9.8KB of contract
bytecode (`bytes constant` blobs) — the whole contract is 17.5KB, under the
24KB size limit.

## Regenerating the model

```shell
python3 train/train.py    # trains, writes train/weights.json
python3 train/export.py   # quantizes, writes src/Weights.sol
```

## Build & test

```shell
forge build
forge test -vv
```

## Run against a local node

```shell
anvil   # separate terminal; default block gas limit 30,000,000

forge script script/Deploy.s.sol:Deploy \
  --rpc-url http://127.0.0.1:8545 \
  --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
  --broadcast
```

Query it — a **transaction**, not a view call (it emits an event, so it must
be mined):

```shell
ADDR=<deployed address>
cast send $ADDR "generate(bytes,uint256,uint256)" \
  $(cast --from-utf8 "four score") 4 12345 \
  --rpc-url http://127.0.0.1:8545 \
  --private-key 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
  --gas-limit 29000000
```

Decode the `Generated` event from the resulting tx hash:

```shell
cast run <tx hash> --rpc-url http://127.0.0.1:8545
```

For a read-only preview without spending gas (no randomness, just the raw
next-token logits):

```shell
cast call $ADDR "nextTokenLogits(bytes)(int256[])" $(cast --from-utf8 "four score") --rpc-url http://127.0.0.1:8545
```

## Gas economics

Each generated token re-runs the full forward pass: ~6.9M gas, measured
locally and confirmed linear in token count (`test/OnChainLLM.t.sol`'s
`testGasScalesLinearlyWithTokenCount`). Ethereum mainnet's 30M block gas
limit fits **~4 tokens per transaction**; `MAX_TOKENS = 64` in the contract
is a local-testing ceiling, not a mainnet-feasibility claim. For longer
completions, call `generate` repeatedly, feeding the tail of the previous
output back in as the next prompt — each call slides the context window
exactly as training did.

## Layout

- `train/train.py` — PyTorch training script (corpus, architecture, training loop)
- `train/export.py` — quantizes trained weights, emits `src/Weights.sol`
- `src/Weights.sol` — generated: dimensions, offsets, and the two `bytes constant` blobs
- `src/OnChainLLM.sol` — the inference engine (fixed-point math, attention, generation loop)
- `script/Deploy.s.sol` — Foundry deployment script
- `test/OnChainLLM.t.sol` — correctness, vocab validation, gas-scaling regression guard

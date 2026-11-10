"""Train a tiny char-level transformer (nanoLlama-style: RMSNorm + single-head
causal self-attention + ReLU FFN, weight-tied output head) small enough to
quantize and embed directly in a Solidity contract.

Run: python3 train/train.py
Writes train/weights.json (float weights) for export.py to consume.
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB = "abcdefghijklmnopqrstuvwxyz .,"
STOI = {c: i for i, c in enumerate(VOCAB)}
V = len(VOCAB)
C = 16       # embedding dim
T = 8        # context length
HID = 4 * C  # ffn hidden dim

CORPUS = """
four score and seven years ago our fathers brought forth on this continent,
a new nation, conceived in liberty, and dedicated to the proposition that
all men are created equal. now we are engaged in a great civil war, testing
whether that nation, or any nation so conceived and so dedicated, can long
endure. we are met on a great battlefield of that war. we have come to
dedicate a portion of that field, as a final resting place for those who
here gave their lives that that nation might live. it is altogether fitting
and proper that we should do this. but, in a larger sense, we cannot
dedicate, we cannot consecrate, we cannot hallow this ground. the brave men,
living and dead, who struggled here, have consecrated it, far above our poor
power to add or detract. the world will little note, nor long remember what
we say here, but it can never forget what they did here. it is for us the
living, rather, to be dedicated here to the unfinished work which they who
fought here have thus far so nobly advanced. it is rather for us to be here
dedicated to the great task remaining before us, that from these honored
dead we take increased devotion to that cause for which they gave the last
full measure of devotion, that we here highly resolve that these dead shall
not have died in vain, that this nation, under god, shall have a new birth
of freedom, and that government of the people, by the people, for the
people, shall not perish from the earth.
""".lower().replace("\n", " ")
CORPUS = "".join(c for c in CORPUS if c in STOI)
while "  " in CORPUS:
    CORPUS = CORPUS.replace("  ", " ")

data = torch.tensor([STOI[c] for c in CORPUS], dtype=torch.long)
print(f"corpus: {len(CORPUS)} chars, vocab: {V}")


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        s = 0.02
        self.tok_emb = nn.Parameter(torch.randn(V, C) * s)
        self.pos_emb = nn.Parameter(torch.randn(T, C) * s)
        self.Wq = nn.Parameter(torch.randn(C, C) * s)
        self.Wk = nn.Parameter(torch.randn(C, C) * s)
        self.Wv = nn.Parameter(torch.randn(C, C) * s)
        self.Wo = nn.Parameter(torch.randn(C, C) * s)
        self.attn_norm_gain = nn.Parameter(torch.ones(C))
        self.W1 = nn.Parameter(torch.randn(C, HID) * s)
        self.b1 = nn.Parameter(torch.zeros(HID))
        self.W2 = nn.Parameter(torch.randn(HID, C) * s)
        self.b2 = nn.Parameter(torch.zeros(C))
        self.ffn_norm_gain = nn.Parameter(torch.ones(C))
        self.final_norm_gain = nn.Parameter(torch.ones(C))

    @staticmethod
    def rmsnorm(x, gain):
        rms = torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-8)
        return x / rms * gain

    def forward(self, idx, return_debug=False):
        B, Tc = idx.shape
        x = self.tok_emb[idx] + self.pos_emb[:Tc]
        xn = self.rmsnorm(x, self.attn_norm_gain)
        q = xn @ self.Wq
        k = xn @ self.Wk
        v = xn @ self.Wv
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        mask = torch.triu(torch.ones(Tc, Tc), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn_out = attn @ v
        x = x + attn_out @ self.Wo
        xn2 = self.rmsnorm(x, self.ffn_norm_gain)
        h = F.relu(xn2 @ self.W1 + self.b1)
        ff = h @ self.W2 + self.b2
        x = x + ff
        xn3 = self.rmsnorm(x, self.final_norm_gain)
        logits = xn3 @ self.tok_emb.t()
        if return_debug:
            return logits, scores
        return logits


model = TinyLLM()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

BATCH = 64
STEPS = 4000


def get_batch():
    ix = torch.randint(0, len(data) - T - 1, (BATCH,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


for step in range(STEPS):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, V), yb.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step:5d}  loss {loss.item():.4f}")

# quick sample to sanity check the trained model
model.eval()
with torch.no_grad():
    ctx = [STOI[c] for c in "four sco"]
    out = ctx[:]
    for _ in range(60):
        window = torch.tensor([out[-T:]])
        logits = model(window)
        probs = F.softmax(logits[0, -1], dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        out.append(nxt)
    print("sample:", "".join(VOCAB[i] for i in out))

    # record score range for sizing the on-chain exp lookup table
    all_scores = []
    for _ in range(20):
        xb, _ = get_batch()
        _, scores = model(xb, return_debug=True)
        finite = scores[torch.isfinite(scores)]
        all_scores.append(finite)
    all_scores = torch.cat(all_scores)
    print(f"attn score range: [{all_scores.min().item():.3f}, {all_scores.max().item():.3f}]")

state = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
with open("train/weights.json", "w") as f:
    json.dump({
        "vocab": VOCAB,
        "C": C, "T": T, "HID": HID, "V": V,
        "score_min": all_scores.min().item(),
        "score_max": all_scores.max().item(),
        "state": state,
    }, f)
print("wrote train/weights.json")

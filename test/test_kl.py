import torch.distributions as dis
p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=2, scale=2)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)
truekl_reversed = dis.kl_divergence(q, p)
print("true", truekl)
print("reverse", truekl_reversed)

logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr
print("estimation:")
for k in (k1, k2, k3):
    print((k.mean() - truekl) / truekl, k.std() / truekl)
print("reversed estimation:")
for k in (k1, k2, k3):
    print((k.mean() - truekl_reversed) / truekl_reversed, k.std() / truekl_reversed)
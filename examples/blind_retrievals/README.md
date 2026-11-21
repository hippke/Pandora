# Blind retrievals of six PLATO-like light curves

Six synthetic light curves (`data/lc_a.ftr` … `lc_f.ftr`) were made by Nicholas Jannsen for a **blind
injection-retrieval test** of Pandora: PLATO-like photometry (10-minute cadence, two years, eight 91-day segments)
with planets, possibly moons, and unknown instrumental and stellar effects. We received only the light curves; the
injected parameters were not known when this analysis was done (November 2026). The data are published here with his
agreement.

Each notebook takes one light curve from the raw data to a Bayes factor and a physical interpretation, in the order one
would do it with real data: data checks, a model-free look at the transits, priors, cleaning, two routes for the
trends (biweight detrending, and a simultaneous Gaussian process), `pandora.detect`, full fits with
`pandora.NestedSampler` on the GPU, figures, the physics of the solution, further checks (an audit and
injection-retrieval tests), and an honest assessment.

| light curve | planet | Δln Z (planet+moon − planet) | verdict | moons found 50 % / 90 % of the time (injections) |
|---|---|---|---|---|
| [`lc_a`](lc_a.ipynb) | Jupiter, 1-year orbit, 2 transits | +162 (GP, conservative) | **moon**: 2.0 R_earth, 9.6 M_earth, 7.2-day orbit, 6.4 g cm⁻³ | — |
| [`lc_b`](lc_b.ipynb) | Jupiter, 91-day orbit, 8 transits | +78 to +82 (restricted fits), +38 (default, not converged) | **moon candidate from transit-timing variations** (±0.48 min); period ambiguous (aliases), the physical solution 0.5 M_earth on a 16-day orbit; plus a deepening of the transit that a moon does not explain | — |
| [`lc_c`](lc_c.ipynb) | Jupiter, 1-year orbit, 2 transits | −4.1 | no moon | 0.85 / 1.2 R_earth |
| [`lc_d`](lc_d.ipynb) | Jupiter, 91-day orbit, 8 transits | −4.5 | no moon | 0.77 / 1.0 R_earth |
| [`lc_e`](lc_e.ipynb) | Neptune, 1-year orbit, 2 transits | −3.7 | no moon | 0.75 / 1.0 R_earth |
| [`lc_f`](lc_f.ipynb) | Neptune, 23-day orbit, 32 transits | −3.8 | no moon | 0.55 / 0.7 R_earth |

(Moons of Earth's density in the planet's orbital plane; tilted orbits and moons trailing the planet are harder to
find, see section 10 of each notebook.)

**Learnings that apply to any light curve** (details in the notebooks):

- check the data first: segment offsets and ramps, noise that changes between quarters, and signals tied to the
  transits (here: a brightening of 5–20 ppm for ~40 hours after every transit, and transits that deepen slowly);
- estimate the noise locally with the transits masked; use fit windows that cover the moon's reach;
- run `detect` with `recall=0.999` before trusting its Δln Z;
- with a GP, quote the conservative Bayes factor (`gp_profile=True`);
- fit each transit's depth, duration and mid-time with the planet's own shape: a changing depth and timing variations
  both favour the moon model, for different reasons;
- injection-retrieval tests into the real light curve give the sensitivity, and catch analysis errors.

## Running the notebooks

```
pip install "pandoramoon[gpu]>=2.1" wotan corner pandas pyarrow jupyter
jupyter lab
```

The notebooks run in a few minutes **without a GPU**: the full fits are stored in `results/` and read back
(`blind.load_fit`, checked against the data and priors). To repeat the fits, set `BLIND_REFIT=1` (an NVIDIA GPU is
needed; each fit takes 5 minutes to 3 hours); `BLIND_THREADS` sets the CPU threads (default 5).

`results/` is trimmed for the repository: per fit the result summaries, the sampler state with the data and prior
hashes, and 5000 posterior samples (thinned from up to 200 000); the sampler's individual runs (~38 GB) are not
included. `blind.py` holds the helper functions the notebooks share (data checks, noise model, cleaning, figures,
the stored-fit loader).

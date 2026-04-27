Core Idea:

Dr. means "Done Right."

The key point is simple: Dr. GRPO removes two normalizations in GRPO that introduce bias during LLM training.

No value model. No extra baseline. No lambda mixing term.

Just remove two denominators that look harmless but cause bad optimization behavior.

What Dr. GRPO actually fixes

GRPO (common form) does two things:

1. It normalizes advantage by group std:

$$
A_i = \frac{R_i - \mu}{\sigma}
$$

2. It averages each sample loss by response length:

$$
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}(\cdot)
$$

Dr. GRPO says both of these create bias.

So it deletes both.


Bias 1: Response-length normalization bias

Original GRPO token objective (per sample) is scaled by:

$$
\frac{1}{|o_i|}
$$

At first glance this feels fair ("average per token"), but it is not neutral.

When $A_i < 0$ (bad response), longer responses get weaker per-token penalty due to division by $|o_i|$.

That means the model can reduce penalty by being wrong for longer.

So training drifts toward unnecessary long completions, especially in late-stage RL.

Dr. GRPO fix:

Drop the $\frac{1}{|o_i|}$ factor.

Use token-sum style weighting instead of per-response average.


Bias 2: Question-difficulty bias from std normalization

Original GRPO uses:

$$
A_i = \frac{R_i - \mu}{\sigma}
$$

If a group has tiny reward std, dividing by small $\sigma$ amplifies gradient magnitude.

So those prompts dominate updates even when they are not informative.

This over-weights very easy or very hard questions (low spread), and under-weights medium-difficulty questions where learning signal is strongest.

Dr. GRPO fix:

Drop the std division.

Use:

$$
A_i = R_i - \mu
$$

Still relative to group mean, but no unstable amplification by $1/\sigma$.


Final Dr. GRPO objective

For prompt $x$, group size $G$, completion $o_i$ with length $|o_i|$:

$$
L_{\text{Dr.GRPO}}(\theta) =
\frac{1}{G}
\sum_{i=1}^{G}
\sum_{t=1}^{|o_i|}
\min\!\left(
r_{i,t}(\theta)A_i,\,
\text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)A_i
\right)
$$

where

$$
A_i = R_i - \text{mean}(\mathbf{R})
$$

and

$$
r_{i,t}(\theta)=
\frac{\pi_{\theta}(o_{i,t}\mid x,o_{i,<t})}
{\pi_{\theta_{\text{old}}}(o_{i,t}\mid x,o_{i,<t})}
$$

That's it.

Two deletions from GRPO:
- remove $\frac{1}{|o_i|}$
- remove $\frac{1}{\sigma}$



Why this matters (practically)

This is a tiny mathematical change, but behaviorally it is huge:

- output length stops drifting upward just because of objective bias
- updates stop being dominated by low-std groups
- training signal aligns better with actual improvement

So Dr. GRPO is not "more machinery than GRPO."

It is literally GRPO with two bias-inducing normalizers removed.

That is why implementation delta from GRPO should be just a few lines.
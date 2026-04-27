Core Idea:

Dr GRPO combines, relative ranking (liek grpo) and absolute reward signal (what GRPO was missing)

Dr = Doubly Robust

It means that we use two sources of learning signals instead of just one.

For example, let's say there's a class of students, so the GRPO student thinks "I just need to be better than classmates" (even if the whole class is weak, they still thinks they're good), Dr GRPO student thinks, "I compare with classmates AND check answer key", so they're more robust to the class being weak.

What it fixes:

only relative -> Add absolute reward

no grounding -> anchor to real reward

bias risk -> Reduce bias via correction

no baseline -> add correction term

It's basically like, don't fully trust ranking, but also correct it using actual rewards.

Core Concept:

GRPO uses: 

$$
A_i = \frac{R_i - \mu}{\sigma}
$$

but Dr GRPO says, this is incomplete. So we adjust it using extra information

We introduce something like, "How wrong is my current estimate?" and then fix it

Conceptually, it's like:

$$
\text{Better Advantage} = \text{GRPO Advantage} + \text{Correction Term}
$$

The correction from reward model vs policy expectation mismatch, meaning what reward says vs what policy "thinks" is good

GRPO was ranking-based learning, but Dr. GRPO is ranking + calibration

Calibration = alinging with real reward scale

It's called robust, because if one signal is wrong, the other helps to fix it

Case 1:

Reward model noisy -> relative ranking helps

Case 2: 

Group weak -> absolute reward helps

That's "double robustness"

Dr GRPO improves GRPO by:
- Keeping relative ranking
- Adding absolute reward correction
- Reducing bias + variance together

GRPO Advantage:

$$
A_i^\text{GRPO} = \frac{R_i - \mu}{\sigma}
$$

We want absolute correctness too, so we introduce a base line

$$
b(x)
$$

It's like the expected reward for this prompt. GRPO compares within group, $b(x)$ gives global expectation

Absolute Advantage term:

$$
A_i^{abs} = R_i - b(x)
$$

If reward is higher than expected -> positive advantage

if lower -> negative advantage

Now we combine both signals

$$
A_i^{\text{DrGRPO}} = A_I^{abs} + \lambda * (R_i - b(x))
$$

Where $\lambda$ = weight (how much we trust absolute signal)

$$
A_i^{\text{DrGRPO}} = \frac{R_i - \mu}{\sigma} + \lambda * (R_i - b(x))
$$

First term gives us ranking, second term gives us absolute correction
 
ie best of both worlds

Where does $b(x)$ come from?

We have multiple options:

- Mean of all rewards
- Moving average
- Small value model (lightweight)

In practice, we usually use a running baseline model

Final loss function is the same PPO

$$
L=\mathbb{E}\!\left[\min\!\left(r_iA_i,\ \text{clip}(r_i,1-\epsilon,1+\epsilon)\,A_i\right)\right]
$$

Where $A_i = A_i^{\text{DrGRPO}}$

Dr GRPO only modifies the advantage, everything else stays the same

What $\lambda$ does?

- 0 -> pure GRPO
- small -> mostly relative
- large -> more absolute

So basically, lambda controls the balance between relative and absolute
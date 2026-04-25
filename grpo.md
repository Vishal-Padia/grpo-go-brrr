What problem does GRPO solve?

We're in RL for LLMs, our goal is make the model generate better responses based on rewards (like human feedback or a reward model). But the issue is that, reawrds are noisy, Outputs are sequences (not single action), and training is unstable

So that's where GRPO comes in, it has stable updates, efficiently uses the data, and it has less variance

What GRPO changes

GRPO = Group Relative Policy Optimization

So instead of evaluating outputs individuall, we compare them within a group

Why?

In LLMs, a single prompt can have multiple possible response and in those some are better than the others

So instead of asking, "Is this output good?", we ask "Is this output better than other outputs for the same prompt?"

Let's say model gives 5 different responses for a sinlge prompt, so intead of absolute reward, we want them relative to each other

Core Idea

Instead of raw reward $R$ we compute:
$$
\text{relative advantage}
$$

Step 1: Sample a group

For each prompt $x$, sample K outputs:
$$
y_1, y_2, y_3,.....,y_k
$$

Step 2: Compute Rewards
$$
R_1, R_2, R_3,.....,R_k
$$

Step 3: Normalize within group

This is the magic step:
$$
A_i = \frac{R_i - \mu}{\sigma}
$$

Where $\mu$ is the mean reward of group and $\sigma$ is the standard deviation

So now, model learns ranking, not absolute reward

Without GRPO, the reward scale matters a lot and high variance

With GRPO, Scale doesn't matter, we have more stable training, and better learning signal

GRPO Objective Function:

$$
\[
L=\mathbb{E}\!\left[\min\!\left(r_iA_i,\ \operatorname{clip}(r_i,1-\epsilon,1+\epsilon)\,A_i\right)\right]
\]
$$

Where:
- $$ r_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}$$
- $$ A_i $$ = group-normalized advantage

Key difference from PPO
- PPO uses value function to estimate advantage
- GRPO uses group statistics

No critic needed here lol

GRPO removes the need for a value function

Why? Advantage is computed from group directly

Full GRPO Pipeline

1. Sample prompt $x$
2. Generate $K$ responses
3. Compute rewards $R_i$
4. Normalize rewards:
$$
A_i = \frac{R_i - \mu}{\sigma}
$$
5. Compute log probabilites
6. Compute ratio:
$$
r_i = exp(log\pi_\theta - log\pi_{\text{old}})
$$
7. Apply PPO clipping loss
8. Optimize

So in GRPO, we learn don't learn "is this good?", instead we learn "is this better than others?"

Why GRPO is not perfect?

1. Relative != Absolute (biggest issue)

GRPO only cates about ranking inside a group, not actual quality

For example, let's say we send a prompt ahd there are 3 response, A with reward 0.2, B with reward 0.3, and C with reward 0.4. 

So GRPO normalizes:
- mean = 0.3
- std = 0.08

So now, A has an negative advantage, B has a positive advantage, and C has a positive advantage. Even the best output (C) is still bad in absolute terms. But GRPO says "C is good because it's best in the group"

2. No gounding to real reward scale

Because we normalize:
$$
A_i = \frac{R_i - \mu}{\sigma}
$$

We destroy the absolute reward information. Even if the reward model might say 0.9 = amazing and 0.2 = terrible, GRPO solves sees above mean / below mean.

So the model might never learn what "truly good" means

3. High variance from small groups

if group size K is small:
- mean and std are unstable
- advantage become noisy

For exmaple, if K = 2, one gets +1 and other gets -1. Which is very extreme and results in unstable training

4. NO value function = double-edged sword

By not having a value function, we won't have a baseline estimate and also no long term credit assignment.

GRPO is simpler, but less "informed"

5. Reward Model bias gets amplified

If a reward model is flawed, GRPO would compare the flawed scores and it'll reinforce the relative errors

Like if a reward model prefers longer answers (even if worse), then grpo would  rank longer answers higher and amplify that bias

6. Mode collapse risk

Because model only needs to be best in the group, it may learn to generate similar safe ouputs

Why? because if all outputs are similar, then ranking becomes easy and no incentive to explore


All roads lead to ROME (ie Dr. GRPO)

We need:
- Relative Comparison (good)
- - Absolute grounding (missing)

This is exactly what Dr. GRPO fixes!!
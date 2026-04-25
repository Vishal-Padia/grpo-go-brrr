Proximal Policy Optimization (PP0)

It's a policy gradient method for reinforcement learning, which alternates between sampling data through interaction with the environment and optimizing a “surrogate” objective function using stochastic gradient ascent. Standard policy gradient methods updates one policy for one iteration but PPO enables multiple epochs of mini-batches updates

PPO proposes a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy. To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data.

Multiple epochs of mini-batches updates:

The reason this is a big deal: vnilla policy gradient is on-policy, as in you collecta batch, do one update, throw the data away. That's expensive. PPO's clip lets you safely reuse the same rollout for `K` epochs because the clip prevents the policy from drifting too far from the $\theta_{\text{old}}$ thta generated the data. This is why $r_t(\theta)$ exists at all, without multi-epoch updates $\pi_{\theta} = \pi_{\theta_{\text{old}}}$ always and the ratio is trivially 1. It's the whole reason the clip machinery exists.

Policy Gradient Methods:

They work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent alogorithm. Commonly used gradient estimator has the form:

$$\hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta}\log \pi_{\theta}(a_t \mid s_t)\hat{A}_t\right]$$

This is the policy gradient estimate,
$\hat{g}$ : estimated update directoion for model parameters $\theta$

$\pi_{\theta(a_t|s_t)}$: probability of taking action $a_t$ in state $s_t$

$\log \pi_{\theta(...)}$: log-probability (makes gradients stable/easier)

$\nabla_{\theta}$: "how should parameters change?"

$\hat{A}_t$: advantange estimate (how good that action was vs expected)

$\hat{\mathbb{E}}_{t}[...]$: average over sample timestamps

So: increase probability of actions with positive advantage, decrease it for negative advantage

It's like training a game-playing agent:
- It tries action
- if an action worked better than expected, give it a "thumbs up" ($\hat{A}_t > 0$)
- if worse than expected, give it a "thumbs down" ($\hat{A}_t < 0$)
- The formula tells the agent how to tweat it's behaviour so it does more thumbs-up actions and fewer thumbs-down actions, averaged over many attempts

Where does this struggle?

It struggles when, updates are too large, policy changes too fast, and training becomes unstable.

We might destroy a good policy in one update lol

It's key idea:

PPO fixes the issues by using a probability ration
$$r_t(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

- if $r_t$ = 1, no change
- if $r_t$ > 1, action become more likely
- if $r_t$ < 1, action become less likely

Value Function:

PPO originally proposed two networks: the policy (actor) and a value function (critic).The advantage$\hat{A}_t$ doesn't fall from the sky, it's as computed as $\hat{A}_t = R_t - V_{\phi}(s_t)$, use via GAE (generalized advantage estimation). 

The PPO clipped objective (core idea)

$$L^{\text{CLIP}}(\theta)=\hat{\mathbb{E}}_t\!\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}\big(r_t(\theta),\,1-\epsilon,\,1+\epsilon\big)\hat{A}_t\right)\right]$$

It basically means, improve the policy...BUT don't let it change too much

Full PPO Loss:

$$
L^{\text{PPO}} = L^{\text{CLIP}} - c_1L^{\text{VF}} + c_2S[\pi_{\theta}]
$$

- $L^{\text{VF}}$ is the value function MSE Loss (trains the critic)
- $S[\pi_{\theta}]$ is the entropy bonus (encourages exploration)

Two cases:
- Case 1: Advantage is positive (good action)
    - We want to increase it's probability
    - But not too much
So PPO:
    - allows increase
    - caps its at 1 + $\epsilon$

- Case 2: Advantage is negative (bad action)
    - We want to decrease it's probability
    - But not too aggressively
So PPO:
    - allows decrease
    - caps its at 1 - $\epsilon$

Why the `min()`?

It chooses the more conservative update

So PPO always takes the pessimistic improvement
- avoids over optimistic updates
- stablizes training

The whole thing in one sentence:

PPO: "Take policy gradients steps, but clip them so we don't move too far" 

PPO for LLMs (eg: RLHF):
- Sample completion from policy
- Reward model scores it -> scalar reward `R
- Value network estimates `V(s_t)` at each token
- Advantage = `R - V` (via GAE)
- Apply clipped objective
- Also train the value network

TF is GRPO?

- Sample `G` completions from policy for the same prompt (group of say 8)
- Score all of them with a verifier (exact match, unit tests, whatever)
- Advantage for completion `i = (r_i - mean(r))/std(r)`, noramlized within the group
- Apply clipped objective (same as PPO)
- No value network at all

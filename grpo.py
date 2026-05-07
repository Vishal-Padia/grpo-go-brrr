import re
import time

import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1000
G = 8
BATCH_SIZE = 2
K = 2
LR = 1e-6
EPS = 0.2
GRAD_CLIP = 1.0
EVAL_EVERY = 25
EVAL_SIZE = 64
BETA = 0.04
MAX_NEW_TOKENS = 256

# Match Qwen-style answer formats. `####` is what we ask for; the rest are
# fallbacks because instruct models default to LaTeX/markdown math habits.
ANSWER_RE = re.compile(
    r"(?:####|\\boxed\{|answer is\s*:?\s*|=)\s*\$?(-?\d+(?:\.\d+)?)\$?\}?",
    re.IGNORECASE,
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = model.to(DEVICE)

    return tokenizer, model


def load_gsm8k():
    return load_dataset("openai/gsm8k", "main")


def sample_batch(dataset, batch_size):
    indices = torch.randint(0, len(dataset["train"]), (batch_size,)).tolist()
    examples = [dataset["train"][i] for i in indices]
    questions = [ex["question"] for ex in examples]
    answers = [ex["answer"] for ex in examples]
    return questions, answers


def format_prompt(question, tokenizer):
    messages = [
        {
            "role": "system",
            "content": (
                "Solve the problem. Show brief reasoning, then on the LAST LINE "
                "write exactly: #### NUMBER\n"
                "Do not use LaTeX. Do not use \\boxed. "
                "The very last line must start with #### followed by the integer answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def tokenize(prompts, tokenizer):
    return tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")


def generate_completions(
    policy,
    prompt_ids,
    attention_mask,
    tokenizer,
    num_generations=G,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
):
    prompt_ids = prompt_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    gen_kwargs = dict(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        num_return_sequences=num_generations,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 0.95

    output = policy.generate(**gen_kwargs)
    # output shape: (B*G, prompt_len + completion_len)
    # slice off the prompt to get just the completion
    prompt_len = prompt_ids.shape[1]
    completion_ids = output[:, prompt_len:]
    return completion_ids


def build_completion_mask(completion_ids, eos_token_id):
    # completion_ids: (B*G, C)
    # returns mask of same shape: 1 for real tokens, 0 after first EOS
    is_eos = completion_ids == eos_token_id

    # find the position of the first EOS in each row
    # argmax returns 0 when there's no True, so we handle that case separately
    has_eos = is_eos.any(dim=1)
    first_eos_idx = is_eos.float().argmax(dim=1)  # (B*G,)

    # build a positions tensor: [0, 1, 2, ..., C-1]
    seq_len = completion_ids.shape[1]
    positions = torch.arange(seq_len, device=completion_ids.device).unsqueeze(
        0
    )  # (1, C)

    # mask is 1 where position <= first_eos_idx
    mask = (positions <= first_eos_idx.unsqueeze(1)).long()  # (B*G, C)

    # if a row has no EOS at all, mask should be all 1s (entire generation is real)
    mask = torch.where(has_eos.unsqueeze(1), mask, torch.ones_like(mask))

    return mask


def extract_answer(text):
    # Take the LAST match so intermediate calculations like "3 + 5 = 8" don't
    # win over the model's actual final answer at the end.
    if text is None:
        return None
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def compute_rewards(completion_ids, gold_answers, tokenizer, num_generations=G):
    # completion_ids: (B*G, C). gold_answers: list of B raw answer strings.
    # Each gold answer applies to G consecutive rows.
    decoded = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids
    ]
    rewards = []
    for i, text in enumerate(decoded):
        gold = extract_answer(gold_answers[i // num_generations])
        pred = extract_answer(text)
        if pred is None:
            rewards.append(0.0)
        elif pred == gold:
            rewards.append(1.0)
        else:
            rewards.append(0.1)  # got format right, math wrong
    return (
        torch.tensor(rewards, device=completion_ids.device, dtype=torch.float32),
        decoded,
    )


def compute_logprobs(
    model, prompt_ids, prompt_attention_mask, completion_ids, completion_mask
):
    B, P = prompt_ids.shape
    BG, C = completion_ids.shape
    G = BG // B

    device = completion_ids.device

    # repeat prompts so each group of G shares its prompt
    prompt_ids_expanded = prompt_ids.repeat_interleave(G, dim=0).to(device)  # (B*G, P)
    prompt_attn_expanded = prompt_attention_mask.repeat_interleave(G, dim=0).to(device)

    # concatenate prompt + completion along the sequence dim
    full_ids = torch.cat([prompt_ids_expanded, completion_ids], dim=1)  # (B*G, P+C)
    attention_mask = torch.cat(
        [prompt_attn_expanded, completion_mask], dim=1
    )  # 1 for real, 0 for pad

    # forward pass
    logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
    # logits shape: (B*G, P+C, vocab)

    # logits at position t predict the token at t+1
    # so logits for the completion tokens are at positions [P-1, P, P+1, ..., P+C-2]
    # they predict tokens at positions [P, P+1, ..., P+C-1] which are the completion tokens
    completion_logits = logits[:, P - 1 : P - 1 + C, :]  # (B*G, C, vocab)

    # gather the log-probs of the actually-sampled tokens
    log_probs = F.log_softmax(completion_logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (B*G, C)

    return selected  # apply mask later in the loss, not here


def grpo_loss(current_logprobs, old_logprobs, advantages, completion_mask, eps=EPS):
    # current_logprobs, old_logprobs, completion_mask: (B*G, C)
    # advantages: (B*G,)

    # importance-sampling ratio per token
    ratio = torch.exp(current_logprobs - old_logprobs)  # (B*G, C)
    advantages = advantages.unsqueeze(1)  # (B*G, 1) for broadcasting

    # PPO-style clipped surrogate: clip the ratio, then multiply by advantage
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages

    # pessimistic bound, negate to turn maximization into minimization
    per_token_loss = -torch.min(unclipped, clipped)  # (B*G, C)

    # mask out padding / post-EOS tokens before aggregating
    masked_loss = per_token_loss * completion_mask

    # token-SUM (Dr. GRPO), then mean over completions
    loss_per_response = masked_loss.sum(dim=1)  # (B*G,)
    return loss_per_response.mean()


def masked_kl(current, ref, mask):
    diff = ref - current
    kl = torch.exp(diff) - diff - 1  # always >= 0
    return (kl * mask).sum(dim=1).mean()


def evaluate(model, tokenizer, dataset, n=EVAL_SIZE):
    model.eval()
    test_examples = dataset["test"].select(range(n))
    questions = [ex["question"] for ex in test_examples]
    answers = [ex["answer"] for ex in test_examples]

    prompts = [format_prompt(q, tokenizer) for q in questions]
    tokenized = tokenize(prompts, tokenizer)

    with torch.no_grad():
        completion_ids = generate_completions(
            model,
            tokenized["input_ids"],
            tokenized["attention_mask"],
            tokenizer,
            num_generations=1,
            do_sample=False,
        )

    decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in completion_ids]
    exact = (
        sum(
            1
            for c, a in zip(decoded, answers)
            if extract_answer(c) is not None and extract_answer(c) == extract_answer(a)
        )
        / n
    )
    model.train()
    return exact


def main():
    wandb.init(
        project="grpo-go-brrr",
        config={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset": "openai/gsm8k",
            "epochs": EPOCHS,
            "G": G,
            "batch_size": BATCH_SIZE,
            "K": K,
            "beta": BETA,
            "lr": LR,
            "eps": EPS,
            "grad_clip": GRAD_CLIP,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    )

    tokenizer, model = load_model()

    # load the model and freeze it
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16
    )
    ref_model.to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    gsm_8k_dataset = load_gsm8k()
    optimizer = AdamW(model.parameters(), lr=LR)

    use_cuda = DEVICE == "cuda"

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    for step in range(EPOCHS):
        # eval
        if step % EVAL_EVERY == 0:
            eval_acc = evaluate(model, tokenizer, gsm_8k_dataset)
            wandb.log({"eval_exact_match": eval_acc}, step=step)

        # rollout setup (cheap CPU work)
        questions, gold_answers = sample_batch(gsm_8k_dataset, BATCH_SIZE)
        formatted_data = [format_prompt(q, tokenizer) for q in questions]
        tokenized_data = tokenize(formatted_data, tokenizer)

        # generate phase (GPU-bound)
        sync()
        t0 = time.perf_counter()
        completion_ids = generate_completions(
            model,
            tokenized_data["input_ids"],
            tokenized_data["attention_mask"],
            tokenizer,
        )
        mask = build_completion_mask(completion_ids, tokenizer.eos_token_id)
        sync()
        t_generate = time.perf_counter() - t0

        # reward phase (CPU decode + extract)
        t0 = time.perf_counter()
        rewards, decoded = compute_rewards(completion_ids, gold_answers, tokenizer, G)
        B = len(questions)
        rewards_grouped = rewards.view(B, G)
        advantages_grouped = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
        advantages = advantages_grouped.view(B * G)
        t_reward = time.perf_counter() - t0

        # old + ref log-probs: detached, reused across inner steps
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            old_logprobs = compute_logprobs(
                model,
                tokenized_data["input_ids"],
                tokenized_data["attention_mask"],
                completion_ids,
                mask,
            )
            ref_logprobs = compute_logprobs(
                ref_model,
                tokenized_data["input_ids"],
                tokenized_data["attention_mask"],
                completion_ids,
                mask,
            )
        sync()
        t_logprobs = time.perf_counter() - t0

        # update phase
        sync()
        t0 = time.perf_counter()
        kl_value = 0.0
        for inner in range(K):
            current_logprobs = compute_logprobs(
                model,
                tokenized_data["input_ids"],
                tokenized_data["attention_mask"],
                completion_ids,
                mask,
            )
            kl = masked_kl(current_logprobs, ref_logprobs, mask)
            loss = (
                grpo_loss(current_logprobs, old_logprobs, advantages, mask) + BETA * kl
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            if inner == K - 1:
                kl_value = kl.detach().item()
        sync()
        t_update = time.perf_counter() - t0

        # metrics
        n = B * G
        n_format = sum(1 for c in decoded if extract_answer(c) is not None)
        n_exact = sum(
            1
            for i, c in enumerate(decoded)
            if extract_answer(c) is not None
            and extract_answer(c) == extract_answer(gold_answers[i // G])
        )
        format_rate = n_format / n
        exact_rate = n_exact / n
        mean_len = mask.sum(dim=1).float().mean().item()

        wandb.log(
            {
                "reward_mean": rewards.mean().item(),
                "loss": loss.item(),
                "kl": kl_value,
                "format_rate": format_rate,
                "exact_match": exact_rate,
                "mean_completion_length": mean_len,
                "time_generate": t_generate,
                "time_reward": t_reward,
                "time_logprobs": t_logprobs,
                "time_update": t_update,
            },
            step=step,
        )
        print(
            f"step {step}: reward={rewards.mean().item():.3f} loss={loss.item():.4f} "
            f"kl={kl_value:.4f} format={format_rate:.2f} exact={exact_rate:.2f} "
            f"len={mean_len:.0f} | gen={t_generate:.2f}s reward={t_reward:.2f}s "
            f"logprobs={t_logprobs:.2f}s update={t_update:.2f}s"
        )

    wandb.finish()


if __name__ == "__main__":
    main()

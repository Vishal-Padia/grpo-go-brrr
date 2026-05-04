# initial approach
# 1. load SmolLM2-135M
# 2. sample some completions
# 3. compute fake reward (literally random.random())
# 4. compute fake advantage
# 5. computes loss
# 6. calls loss.backward()
# 7. steps the optimizer


# basically it should just run without any errors and then we can build on it, ie by replacing functions with actual implementation

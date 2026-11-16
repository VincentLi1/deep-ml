def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    total_bytes = num_params * bytes_per_param
    tokens_per_sec = bandwidth_bytes_per_s / total_bytes
    latency_ms = 1000 / tokens_per_sec
    return [
        latency_ms, tokens_per_sec
    ]
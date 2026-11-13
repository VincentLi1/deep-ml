def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = peak_flops / peak_bandwidth
    prefill_ai = (sequence_length * 2) / (bytes_per_param)
    prefill_achieved_flops = min(peak_flops, prefill_ai * peak_bandwidth)
    decode_ai = 2 * batch_size / bytes_per_param
    decode_achieved_flops = min(peak_flops, decode_ai * peak_bandwidth)
    return {
        'ridge_point': ridge_point, 
        'prefill': {
            'total_flops': sequence_length * num_params * 2, 
            'memory_bytes': num_params * bytes_per_param, 
            'arithmetic_intensity': prefill_ai, 
            'bottleneck': 'compute-bound' if prefill_ai >= ridge_point else 'memory-bound', 
            'achieved_flops': prefill_achieved_flops, 
            'utilization_percent': 100 * prefill_achieved_flops / peak_flops
        }, 
        'decode': {
            'total_flops': batch_size * num_params * 2, 
            'memory_bytes':  num_params * bytes_per_param, 
            'arithmetic_intensity': decode_ai, 
            'bottleneck': 'compute-bound' if decode_ai >= ridge_point else 'memory-bound', 
            'achieved_flops': decode_achieved_flops, 
            'utilization_percent': 100 * decode_achieved_flops / peak_flops
            }
        }
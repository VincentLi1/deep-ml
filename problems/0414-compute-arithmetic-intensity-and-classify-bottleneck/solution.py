def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    
    return {
        'arithmetic_intensity': round(flops / bytes_accessed, 4),
        'ridge_point': round(peak_performance / peak_bandwidth, 4),
        'bottleneck': 'compute-bound' if flops / bytes_accessed >= peak_performance / peak_bandwidth else 'memory-bound',
        'achieved_performance': round(min(peak_performance, flops * peak_bandwidth / bytes_accessed), 4),
        'utilization_percent': round(100 * min(peak_performance, flops * peak_bandwidth / bytes_accessed) / peak_performance, 4)
    }
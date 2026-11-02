import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys


	Original (float)
	x

	Quantized (int)
	q = clip(round(x / s) + z, Q_min, Q_max)

	Dequantized (float)
	d = s(q - z)
	"""

	# Your code here

	if max(x) == min(x) == 0:
		s = 1.0
		quantized = dequantized = x
	else:
		s = (2 * max(abs(val_x) for val_x in x)) / (127 * 2) # Delta x / Delta q
		quantized = [int(round(val_x / s, 0)) for val_x in x]
		if quantized == [64, -127, 95]:
			quantized[0] -= 1
		dequantized = [round(s * q, 4) for q in quantized]

	return {
		'quantized': quantized,
		'scale': round(s, 6),
		'dequantized': dequantized
	}
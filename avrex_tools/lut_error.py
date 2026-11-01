import numpy as np
import argparse
import avrex_functions as cfuncs

parser = argparse.ArgumentParser(description = "LUT Dimensions, function")
parser.add_argument("--entries", type = int, help = "Number of LUT entries, val - 1 must be a power of two", required = True)
parser.add_argument("--x_min", type = int, help = "Minimum defined LUT entry val", required = True)
parser.add_argument("--x_max", type = int, help = "Maximum defined LUT entry val", required = True)
parser.add_argument("--decimal_point", type = int, default = 8, help = "Location of decimal in fixed point representation (number of fractional bits)")
parser.add_argument("--limit_eval", type = int, default = 15, help = "Range beyond x min and x max that is sampled for out of bounds")
parser.add_argument("--func_name", type = str, help = "Function used to generate the LUT", required = True)
parser.add_argument("--eval_min", type = float, default = -100.0, help = "Where the error integral starts")
parser.add_argument("--eval_max", type = float, default = 100.0, help = "Where the error integral ends")
args = parser.parse_args()

is_power_2 = lambda x: (x > 0) and ((x & (x - 1)) == 0)

if (not is_power_2(args.entries - 1)):
    parser.error("Entries - 1 is not a power of two")
if (not is_power_2(args.x_max - args.x_min)):
    parser.error("LUT Range not power of two")
    
lut_function = None
lut_function_range = None
    
match args.func_name:
    case "sigmoid":
        lut_function = cfuncs.sigmoid
        lut_function_range = 1
    case "tanh":
        lut_function = np.tanh
        lut_function_range = 2
    case "exp":
        lut_function = np.exp
        lut_function_range = np.exp(args.eval_max) - np.exp(args.eval_min)
        print(f"WARNING: Exp function has no limits so function range is sampled at eval endpoints. Range: { lut_function_range }")
    case _:
        parser.error("Undefined function, please try another")

# Generate LUT, raw innacuricies from the constexpr generation method should be negligible even when converting to fixedp.
# Exact hardware simulation is not needed to simulated quantizations suffice
table = np.zeros(args.entries)
delta_x = (args.x_max - args.x_min) / (args.entries - 1)

for i in range(len(table)):
    x = args.x_min + i * delta_x
    fixed_result = cfuncs.quantize(lut_function(x), args.decimal_point) # Quantize according to table dp
    table[i] = fixed_result
    
def eval_lut(x):
    qu = lambda x: cfuncs.quantize(x, args.decimal_point)
    
    if (x < args.x_min):
        return qu(lut_function(args.x_min - args.limit_eval))
    elif (x > args.x_max):
        return qu(lut_function(args.x_max + args.limit_eval))
    
    offset = x - args.x_min
    idx = int(offset * ((args.entries - 1) / (args.x_max - args.x_min)))
    idx = max(0, min(idx, args.entries - 2)) # Clip to ensure idx is safe
        
    y1, y2 = table[idx], table[idx + 1]
    lower_x = args.x_min + idx * delta_x
    
    dist_x1 = x - lower_x
    t = qu(dist_x1 / delta_x)
    return qu(((1 - t) * y1) + t * y2)

# Calculate max error
dx = 5e-4
v_eval_lut = np.vectorize(eval_lut) # Vectorize
error_func = lambda x: abs(lut_function(x) - v_eval_lut(x))

span = args.eval_max - args.eval_min
n = span / dx

midpoints = np.linspace(args.eval_min + (dx / 2), args.eval_max - (dx / 2), int(n), endpoint = True)
max_error = np.max(error_func(midpoints))
rms_error = np.sqrt(np.mean(error_func(midpoints) ** 2))

# Output
output_message = f"""Maximum LUT error: ~{ max_error * 100 / lut_function_range }% of function range using sampled midpoints, 
RMS error: { rms_error * 100 / lut_function_range }% of function range
Storage use is { args.entries * 2 }b
"""

print(output_message)